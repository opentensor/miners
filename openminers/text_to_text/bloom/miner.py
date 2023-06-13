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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import deepspeed
import os

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

class BloomChatMiner( openminers.BasePromptingMiner ):

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        pass

    def __init__( self, *args, **kwargs):
        super( BloomChatMiner, self ).__init__( *args, **kwargs )
        model_name = "sambanovasystems/BLOOMChat-176B-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 )
        self.pipe = pipeline( "text-generation", self.model, tokenizer=tokenizer, device = local_rank, max_new_tokens = 256 )
        self.pipe.model = deepspeed.init_inference(self.pipe.model,
                                        mp_size=world_size,
                                        dtype=getattr(torch, int8),
                                        replace_with_kernel_inject=True)

    @staticmethod
    def _process_history( history: List[ Dict[str, str] ] ) -> str:
        processed_history = ''
        for message in history:
            if message['role'] == 'system':
                processed_history += '<human>: ' + message['content'] + '\n'
            if message['role'] == 'assistant':
                processed_history += '<bot>: ' + message['content'] + '\n'
            if message['role'] == 'user':
                processed_history += '<human>: ' + message['content'] + '\n'
        return processed_history

    def forward( self, messages: List[Dict[str, str]]  ) -> str:
        history = self._process_history(messages)
        resp = self.pipe( history )[0]['generated_text'].split(':')[-1].replace( str( history ), "")
        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print(resp)
        print(resp)
        return resp

if __name__ == "__main__":  
    miner = BloomChatMiner()
    miner.run()
    # with miner:
    #     while True:
    #         print ('running...', time.time())
    #         time.sleep(1)