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

import json
import wandb
import sys
import time
import argparse
import openminers
import bittensor as bt
from tqdm import tqdm
from typing import List, Dict

# from datasets import load_dataset
# DATASET = iter( load_dataset( 'squad_v2', split = 'train', streaming = True ).shuffle( buffer_size = 10000 ))

def get_mock_query( ) -> List[Dict[str, str]]:
    prompt = """you are a chatbot that can come up with unique questions about many things."""
    message = "ask me a random question about anything"
    roles = ['system', 'user']
    messages = [ prompt, message ]
    packed_messages = [ json.dumps({"role": role, "content": message}) for role, message in zip( roles,  messages )]
    return packed_messages, roles, messages

def run():
    # Parse miner class.
    MINER = getattr( openminers, sys.argv[2] )
    N_STEPS = int(sys.argv[3]) 

    # Load miner config
    config = MINER.config()

    # Set mock values on miner.
    config.miner.blacklist.allow_non_registered = True
    config.no_serve_axon = True
    config.no_register = True
    config.no_set_weights = True
    config.wallet._mock = True
    config.axon.external_ip = "127.0.0.1"
    config.axon.port = 9090
    bt.logging.success( f'Running benchmarks for miner: { sys.argv[1] }' )

    # Instantiate the miner axon
    wallet = bt.wallet.mock()
    axon = bt.axon( wallet = wallet, config = config, metagraph = None )
    dendrite = bt.text_prompting( axon = axon.info(), keypair = wallet.hotkey )
    bt.logging.success( f'dendrite: { dendrite }' )

    # Instantiate miner.
    with MINER( config = config, axon = axon, wallet = wallet ) as miner:
        for step in range( N_STEPS ):
            _, roles, messages = get_mock_query()
            dendrite.forward( roles = roles, messages = messages, timeout = 1e6 )

if __name__ == "__main__":
    run()