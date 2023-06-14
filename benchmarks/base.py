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

import os
import json
import time
import torch
import argparse
import openminers
import bittensor as bt
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset
from reward import RewardModel

DATASET = iter( load_dataset( 'squad_v2', split = 'train', streaming = True ).shuffle( buffer_size = 10000 ))


def get_dataset_query( prompt ) -> List[Dict[str, str]]:
    message = next( DATASET )['question']
    roles = ['system', 'user']
    messages = [ prompt, message ]
    packed_messages = [ json.dumps( {"role": role, "content": message} ) for role, message in zip( roles,  messages )]
    return packed_messages, roles, messages

def get_mock_query( ) -> List[Dict[str, str]]:
    prompt = "You are a chatbot that can come up with unique questions about many things."
    message = "ask me a random question about anything"
    roles = ['system', 'user']
    messages = [ prompt, message ]
    packed_messages = [ json.dumps( {"role": role, "content": message} ) for role, message in zip( roles,  messages )]
    return packed_messages, roles, messages

def load_reward_model( device, reward_path: str = None ):
    bt.logging.info( "Loading reward model" )
    reward_model = RewardModel( model_path="EleutherAI/gpt-j-6b", device=device )
    if reward_path:
        for fpath in os.listdir( reward_path ):
            if fpath.endswith( ".pt" ) or fpath.endswith( ".bin" ):
                checkpoint = os.path.join( reward_path, fpath )
                break
        ckpt_state = torch.load( checkpoint )
        reward_model.load_state_dict( ckpt_state )
    reward_model.eval()
    reward_model.half()
    reward_model.requires_grad_( False )
    reward_model.to( device )
    bt.logging.debug( str( reward_model ) )
    return reward_model

def compute_reward( response: str, device: str, reward_model: RewardModel, prompt: str ):
    completions_without_prompt = [ response ]
    rewards = reward_model.reward(
        completions_with_prompt=[ prompt + response ],
        completions_without_prompt=completions_without_prompt,
        difference=True,
        shift=3,
    ).to( device )
    return rewards

def run():
    # Parse args.
    parser = argparse.ArgumentParser()
    parser.add_argument( '--miner', type=str, default="HermesMiner", help='miner class to benchmark' )
    parser.add_argument( '--n_steps', type=int, default=3, help='number of steps to benchmark' )
    parser.add_argument( '--device', type=str, default="cuda", help='device to benchmark on' )
    parser.add_argument( '--use_mock_query', action='store_true', help='use mock query' )
    parser.add_argument( '--system_prompt', type=str, default="A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions. ", help='system prompt' )
    args = parser.parse_args()

    # Parse miner class.
    # TODO: allow miner config arguments to passthrough (instead of defaults)
    MINER = getattr( openminers, args.miner )
    N_STEPS = int( args.n_steps )
    
    # Load reward model (local)
    reward_model = load_reward_model( args.device )

    # Load miner config + merge with BaseMiner config.
    config = MINER.config()
    config.merge( openminers.BaseMiner.config() )

    # Set mock values on miner.
    config.miner.blacklist.allow_non_registered = True
    config.no_serve_axon = True
    config.no_register = True
    config.no_set_weights = True
    config.wallet._mock = True
    config.axon.external_ip = "127.0.0.1"
    config.axon.port = 9090
    bt.logging.success( f'Running benchmarks for miner: { args.miner }' )

    # Instantiate the miner axon
    wallet = bt.wallet.mock()
    axon = bt.axon( wallet = wallet, config = config, metagraph = None )
    dendrite = bt.text_prompting( axon = axon.info(), keypair = wallet.hotkey )
    bt.logging.success( f'Loaded dendrite: { dendrite }' )

    results = { 'elapsed': [], 'reward': [] }

    # Instantiate miner.
    with MINER( config = config, axon = axon, wallet = wallet ) as miner:
        total_time = 0
        total_reward = 0
        for _ in tqdm( range( N_STEPS ) ):

            if args.use_mock_query:
                prompt = "You are a chatbot that can come up with unique questions about many things."
                _, roles, messages = get_mock_query()
            else:
                prompt = args.system_prompt
                _, roles, messages = get_dataset_query( args.system_prompt )

            # Start timing
            start_time = time.time()

            # Forward query
            completion = dendrite.forward( roles = roles, messages = messages, timeout = 1e6 ).completion

            # End timing
            time_taken = time.time() - start_time
            total_time += time_taken

            # Compute reward
            reward = compute_reward( completion, device=args.device, reward_model=reward_model, prompt=prompt )
            total_reward += reward

            # Log results
            results['elapsed'].append( time_taken )
            results['reward'].append( reward )

        average_time = total_time / N_STEPS
        average_reward = total_reward / N_STEPS

    results["average_time"] = average_time
    results["average_reward"] = float( average_reward )
    print("=== Model Results ===")
    print(f"Average Time per Step: {results['average_time']:.2f} seconds")
    print(f"Average Reward per Step: {results['average_reward']:.2f}")

if __name__ == "__main__":
    run()