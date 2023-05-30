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
import argparse
import openminers
import bittensor as bt
from tqdm import tqdm
from typing import List, Dict

# Load miner name
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='Miner Name.', default='TemplateMiner')
config = bt.config( parser )

miner_config = getattr(openminers, config.name).config()
print(miner_config)



# def get_mock_query( ) -> List[Dict[str, str]]:
#     prompt = """this is a mock request"""
#     message = "who am I, really?"
#     roles = ['system', 'user']
#     messages = [ prompt, message ]
#     packed_messages = [ json.dumps({"role": role, "content": message}) for role, message in zip( roles,  messages )]
#     return packed_messages, roles, messages

# # Send single query through miner's axon.
# def benchmark_forward():     
#     wandb.init( project='openminers', name = 'benchmark_forward' )

#     # Create a mock wallet.
#     wallet = bt.wallet().create_if_non_existent()
#     axon = bt.axon( wallet = wallet, port = 9090, ip = "127.0.0.1", metagraph = None )
#     config = openminers.TemplateMiner.config()
#     config.allow_non_registered = True
#     miner = openminers.TemplateMiner( config = config, axon = axon  )
#     miner.axon.start()

#     # Get endpoint.
#     axon_endpoint = axon.info() 
#     axon_endpoint.ip = "127.0.0.1"
#     axon_endpoint.port = 9090
#     dendrite = bt.text_prompting( axon = axon_endpoint, keypair = wallet.hotkey )

#     # Make query.
#     responses = []
#     for step in range(10000):
#         _, roles, messages = get_mock_query()
#         forward_call = dendrite.forward( roles = roles, messages = messages, timeout = 1e6 )
#         responses.append( forward_call )
#         wandb.log( { 'step': step } )

    

# if __name__ == "__main__":
#     benchmark_forward()