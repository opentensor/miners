# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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
import openminers
import bittensor
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class StabilityAIMiner( openminers.BasePromptingMiner):

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument('--stabilityai.api_key', type = str, help='huggingface api key', required = True )
        parser.add_argument('--stabilityai.model_size', type=int, choices=[3, 7], default=7, help='Run the 3B or 7B model.')
        parser.add_argument('--stabilityai.device', type=str, help='Device to load model', default="cuda" )
        parser.add_argument('--stabilityai.suffix', type=str, default=None, help="The suffix that comes after a completion of inserted text.")
        parser.add_argument('--stabilityai.max_tokens', type=int, default=20, help="The maximum number of tokens to generate in the completion.")
        parser.add_argument('--stabilityai.num_return_sequences', type=int, default=1, help='Description of num_return_sequences')
        parser.add_argument('--stabilityai.num_beams', type=int, default=1, help='Description of num_beams')
        parser.add_argument('--stabilityai.do_sample', type=bool, default=True, help='Description of do_sample')
        parser.add_argument('--stabilityai.temperature', type=float, default=1.0, help='Description of temperature')
        parser.add_argument('--stabilityai.top_p', type=float, default=0.95, help='Description of top_p')
        parser.add_argument('--stabilityai.top_k', type=int, default=10, help='Description of top_k')
        parser.add_argument('--stabilityai.stopping_criteria', type=str, default='stop', help='Description of stopping_criteria')

    def __init__( self, api_key: Optional[str] = None, *args, **kwargs):
        super( StabilityAIMiner, self ).__init__( *args, **kwargs )
        bittensor.logging.info( 'Loading togethercomputer/StabilityAI {}B model...'.format( self.config.stabilityai.model_size ) )
        if api_key is None and self.config.stabilityai.api_key is None:
            raise ValueError('the miner requires passing --stabilityai.api_key as an argument of the config or to the constructor.')
        self.model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stablelm-tuned-alpha-{}b".format( self.config.stabilityai.model_size ),
            use_auth_token = api_key or self.config.stabilityai.api_key,
            torch_dtype=torch.float16
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/stablelm-tuned-alpha-{}b".format( self.config.stabilityai.model_size ),
            use_auth_token = api_key or self.config.stabilityai.api_key
        )

        if self.config.stabilityai.device == "cuda":
            self.model = self.model.to( self.config.stabilityai.device )

        self.pipe = pipeline(
            "text-generation",
            self.model,
            tokenizer = self.tokenizer,
            device = 0,
            max_new_tokens = self.config.stabilityai.max_tokens,
            num_return_sequences = self.config.stabilityai.num_return_sequences,
            num_beams = self.config.stabilityai.num_beams,
            do_sample = self.config.stabilityai.do_sample,
            temperature = self.config.stabilityai.temperature,
            top_p = self.config.stabilityai.top_p,
            top_k = self.config.stabilityai.top_k,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )
        bittensor.logging.info( "StabilityAI {}B model loaded".format( self.config.stabilityai.model_size ) )

    @staticmethod
    def _process_history( history: List[ Dict[str, str] ] ) -> str:
        processed_history = ''
        for message in history:
            if message['role'] == 'system':
                processed_history += '<|SYSTEM|>: ' + message['content'] + '\n'
            if message['role'] == 'assistant':
                processed_history += '<|ASSISTANT|>: ' + message['content'] + '\n'
            if message['role'] == 'user':
                processed_history += '<|USER|>: ' + message['content'] + '\n'
        return processed_history

    def forward( self, messages: List[Dict[str, str]]  ) -> str:
        history = self._process_history(messages)
        return self.pipe( history )[0]['generated_text'].split(':')[-1].replace( str( history ), "")

if __name__ == "__main__":  
    with StabilityAIMiner():
        while True:
            print ('running...', time.time())
            time.sleep(1)
