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
import bittensor as bt
from typing import List, Dict
from .miner import TemplateMiner
        
def get_mock_query( ) -> List[Dict[str, str]]:
    prompt = """this is a mock request"""
    message = "who am I, really?"
    roles = ['system', 'user']
    messages = [ prompt, message ]
    packed_messages = [ json.dumps({"role": role, "content": message}) for role, message in zip( roles,  messages )]
    return packed_messages

def test_template_forward():
    config = TemplateMiner.config()
    config.mock_subtensor = True
    miner = TemplateMiner( config = config )

    # Send single query through miner.
    miner.forward( get_mock_query() )


def test_axon_forward():
    config = TemplateMiner.config()
    config.mock_subtensor = True
    miner = TemplateMiner( config = config )

    # Make connection to miner.
    wallet = bt.wallet().create_if_non_existent()
    axon = bt.axon( wallet = wallet, port = 9090, ip = "127.0.0.1", metagraph = None )
    dendrite = bt.text_prompting( axon = axon.info(), keypair = wallet.hotkey )



    # Send single query through miner.
    miner.forward( get_mock_query() )
