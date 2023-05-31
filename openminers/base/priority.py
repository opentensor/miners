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

import wandb
import bittensor as bt
from typing import List, Dict, Union, Tuple, Callable

def default_priroity( self, forward_call: "bt.TextPromptingForwardCall" ) -> float:
    # Check if the key is registered.
    registered = False
    if self.metagraph is not None:
        registered = forward_call.src_hotkey in self.metagraph.hotkeys

    # Non-registered users have a default priority.
    if not registered:
        return self.config.miner.default_priority

    # If the user is registered, it has a UID.
    uid = self.metagraph.hotkeys.index( forward_call.src_hotkey )
    stake_amount = self.metagraph.S[uid].item() 
    return stake_amount

def priority( self, func: Callable, forward_call: "bt.TextPromptingForwardCall" ) -> float:

    # Check to see if the subclass has implemented a priority function.
    try: 

        # Call the subclass priority function and return the result.
        priority = func(forward_call)
    
    except NotImplementedError:
        # The subclass has not implemented a priority function.
        pass

    except Exception as e:
        # An error occured in the subclass priority function.
        bt.logging.error( f'Error in priority function: {e}') 

        # If the subclass has not implemented a priority function, we use the default priority.
        priority = default_priroity(self, forward_call)
    
    finally:
        # Log the priority to wandb.
        if self.config.wandb.on: wandb.log( { 'priority': 0.0, 'hotkey': forward_call.src_hotkey } )

        # Return the priority.
        return priority