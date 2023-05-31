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
import wandb
import random
import bittensor as bt
import traceback
from typing import Callable, Dict, List

def forward( self, func: Callable, messages: List[Dict[str, str]] ) -> str:
    
    start_time = time.time()
    try:
        if random.random() < 0.5:
            raise Exception('Random error')
        else:
            time.sleep( random.random() )
            response = func( messages )
    except Exception as e:
        traceback.print_stack()
        bt.logging.error( f'Error in forward function: { e }')
        end_time = time.time()
        if self.config.wandb.on: wandb.log( { 'success': 0 } )
        return 'Error in forward function'
    
    if self.config.wandb.on: wandb.log( { 'forward': 1, 'resplen': len(response), 'qtime': time.time() - start_time, 'success': 1 } )
    return response