# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import copy
import wandb
import torch
import argparse
import threading
import openminers
import bittensor
 as bt

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple

from .run import run
from .priority import priority
from .blacklist import blacklist
from .mock import MockSubtensor
from .config import config, check_config

class BaseMiner( ABC ):

    @classmethod
    def config( cls ) -> "bt.Config": return config( cls )

    @classmethod
    @abstractmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        ...

    @abstractmethod
    def forward( self, messages: List[Dict[str, str]] ) -> str:
        ...

    def _forward( self, messages: List[Dict[str, str]] ) -> str:
        return self.forward( messages )

    def priority( self, forward_call: "bt.TextPromptingForwardCall" ) -> float:
        return priority( self, forward_call )
    
    def blacklist( self, forward_call: "bt.TextPromptingForwardCall" ) -> Union[ Tuple[bool, str], bool ]:
        return blacklist( self, forward_call )

    def __init__( 
            self, 
            config: "bt.Config" = None,
            axon: "bt.axon" = None,
            wallet: "bt.Wallet" = None,
            subtensor: "bt.Subtensor" = None,
            synapse: "bt.Synapse" = None,
        ):

        # Instantiate and check config.
        self.config = copy.deepcopy( config or BaseMiner.config() )
        check_config( BaseMiner, self.config )

        # Instantiate logging.
        bt.logging( config = self.config, logging_dir = self.config.miner.full_path )

        # Instantiate subtensor.
        if self.config.miner.mock_subtensor:
            self.subtensor = subtensor or MockSubtensor( self.config )
        else:
            self.subtensor = subtensor or bt.subtensor( self.config )

        # Instantiate metagraph.
        self.metagraph = self.subtensor.metagraph( self.config.netuid )

        # Instantiate wallet.
        self.wallet = wallet or bt.wallet( self.config )

        # Instantiate axon.
        self.axon = axon or bt.axon(
            wallet = self.wallet,
            metagraph = self.metagraph,
            config = self.config,
        )

        # Define synapse.
        class Synapse( bt.TextPromptingSynapse ):

            # Build priority function.
            def priority( _, forward_call: "bt.TextPromptingForwardCall" ) -> float:
                return self.priority( forward_call )
            
            # Build blacklist function.
            def blacklist( _, forward_call: "bt.TextPromptingForwardCall" ) -> Union[ Tuple[bool, str], bool ]:
                return self.blacklist( forward_call )

            # Build forward function.
            def forward( _, messages: List[Dict[str, str]] ) -> str:
                return self.forward( messages )      

            # Build backward function.
            # TODO(const): accept this.
            def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: 
                pass
                  
        # Instantiate synapse.
        self.synapse = synapse or Synapse( axon = self.axon )

        # Init wandb.
        if self.config.wandb.on:
            self.wandb = wandb.init(
                project = self.config.wandb.project_name,
                entity = self.config.wandb.entity,
                config = self.config,
                mode = 'offline' if self.config.wandb.offline else 'online',
                dir = self.config.miner.full_path,
                magic = True,
            )

        # Instantiate runners.
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None

    def run( self ): run( self )

    def start(self):
        if not self.is_running:
            bt.logging.debug( f'Starting miner background thread') 
            self.should_exit = False
            self.thread = threading.Thread( target = self.run, daemon = True )
            self.thread.start()
            self.is_running = True
            bt.logging.debug( f'Started') 

    def stop(self):
        if self.is_running:
            bt.logging.debug( f'Stopping miner background thread...') 
            self.should_exit = True
            self.thread.join( 5 )
            bt.logging.debug( f'Stopped') 

    def __enter__( self ): self.start()

    def __exit__( self, exc_type, exc_value, traceback ): self.stop()
