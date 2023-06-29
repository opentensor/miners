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


def forward(self, func: Callable, messages: List[Dict[str, str]]) -> str:
    """ Forwards a list of messages to the miner's forward function."""

    # Run the subclass forward function.
    try:
        start_time = time.time()
        response = func(messages)
        success = 1

    # There was an error in the error function.
    except Exception as e:
        bt.logging.error(f"Error in forward function: { e }")
        response = ""
        success = 0

    finally:
        # Log the response length and qtime.
        if self.config.wandb.on:
            wandb.log(
                {
                    "forward_response_length": len(response),
                    "forward_elapsed": time.time() - start_time,
                    "forward_was_success": success,
                }
            )

        # Return the response.
        return response
