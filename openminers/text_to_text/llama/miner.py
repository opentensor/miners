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
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

class LlamaMiner( openminers.BasePromptingMiner ):

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        pass

    def __init__( self, *args, **kwargs):
        super( LlamaMiner, self ).__init__( *args, **kwargs )
        tokenizer = AutoTokenizer.from_pretrained( "huggyllama/llama-30b" )
        config = AutoConfig.from_pretrained("huggyllama/llama-30b", trust_remote_code=True)
        
        model_hidden_size = config.hidden_size

        # batch size has to be divisible by world_size, but can be bigger than world_size
        train_batch_size = 1 * world_size

        ds_config = {
            "fp16": {
                "enabled": False,
            },
            "bf16": {
                "enabled": False,
            },
            "zero_optimization": {
                "stage": 3,
                # "offload_param": {
                #     "device": "cpu",
                #     "pin_memory": True
                # },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": model_hidden_size * model_hidden_size,
                "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
                "stage3_param_persistence_threshold": 10 * model_hidden_size
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }

        dschf = HfDeepSpeedConfig(ds_config)

        self.model = AutoModelForCausalLM.from_pretrained(  "huggyllama/llama-30b", torch_dtype=torch.float16 )
        ds_engine = deepspeed.initialize(model=self.model,
                                 config_params=ds_config,
                                 model_parameters=None,
                                 optimizer=None,
                                 lr_scheduler=None)[0]
        ds_engine.module.eval() # inference
        self.pipe = pipeline( "text-generation", ds_engine.module, tokenizer=tokenizer, device = 0, max_new_tokens = 256 )


    @staticmethod
    def _process_history( history: List[ Dict[str, str] ] ) -> str:
        processed_history = ''
        for message in history:
            if message['role'] == 'system':
                processed_history += 'system: ' + message['content'] + '\n'
            if message['role'] == 'assistant':
                processed_history += 'assistant: ' + message['content'] + '\n'
            if message['role'] == 'user':
                processed_history += 'user: ' + message['content'] + '\n'
        return processed_history

    def forward( self, messages: List[Dict[str, str]]  ) -> str:
        history = self._process_history(messages)
        resp = self.pipe( history )[0]['generated_text'].split(':')[-1].replace( str( history ), "")
        return resp

if __name__ == "__main__":  
    miner = LlamaMiner()
    miner.run()
    # with miner:
    #     while True:
    #         print ('running...', time.time())
    #         time.sleep(1)