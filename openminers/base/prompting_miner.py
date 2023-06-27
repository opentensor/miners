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

import torch
import argparse
import bittensor as bt

from abc import ABC
from typing import List, Dict, Union, Tuple

from .forward import forward
from .priority import priority
from .blacklist import blacklist
from .miner import BaseMiner


class BasePromptingMiner( BaseMiner, ABC ):

    @classmethod
    def config( cls ) -> "bt.Config":
        parser = argparse.ArgumentParser()
        cls.add_super_args( parser )
        return bt.config( parser )

    @classmethod
    def add_super_args( cls, parser: argparse.ArgumentParser ):
        """ Add arguments specific to BasePromptingMiner to parser.
        """
        cls.add_args(parser)
        parser.add_argument(
            '--neuron.max_batch_size', 
            type = int, 
            help = 'The maximum batch size for forward requests.',
            default = -1
        )
        parser.add_argument(
            '--neuron.max_sequence_len', 
            type = int, 
            help = 'The maximum sequence length for forward requests.',
            default = -1
        )

    def __init__( self, *args, **kwargs ):
        super( BasePromptingMiner, self ).__init__( *args, **kwargs )

        # Define synapse.
        class Synapse( bt.TextPromptingSynapse ):

            # Build priority function.
            def priority( _, forward_call: "bt.SynapseCall" ) -> float:
                return priority( self, self.priority, forward_call )

            # Build blacklist function.
            def blacklist( _, forward_call: "bt.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
                return blacklist( self, self.blacklist, forward_call )

            # Build forward function.
            def forward( _, messages: List[Dict[str, str]] ) -> str:
                return forward( self, self.forward, messages )    

            # Build backward function.
            # TODO(const): accept this.
            def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: 
                pass

        # Instantiate synapse.
        self.synapse = Synapse( axon = self.axon )
