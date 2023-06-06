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
import time
import torch
import argparse
import bittensor
import openminers
from typing import List, Union, Tuple


class TemplateEmbeddingMiner( openminers.BaseEmbeddingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def config( cls ) -> "bittensor.Config":
        parser = argparse.ArgumentParser( description='Template Embedding Miner Configs' )
        cls.add_args( parser )
        return bittensor.config( parser )
    
    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--template_arg', default='template_default', type=str, help='Template argument.' )

    def __init__( self, *args, **kwargs ):
        super( TemplateEmbeddingMiner ).__init__( *args, **kwargs )
        self.template_arg = self.config.miner.template_arg

    def blacklist( self, forward_call: "bittensor.TextToEmbeddingForwardCall" ) -> Union[ Tuple[bool, str], bool ]:
        return False
    
    def priority( self, forward_call: "bittensor.TextToEmbeddingForwardCall" ) -> float:
        return 0.0        

    def forward( self, text: List[str] ):
        return torch.zeros( ( self.config.miner.embedding_size ) )


with TemplateEmbeddingMiner(): 
    while True: 
        time.sleep( 1 )

