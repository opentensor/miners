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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
import bittensor
import deepspeed
import os
# from openbase.config import config

deployment_framework = "ds_inference"
deployment_framework = "accelerate"

class LlamaMiner( openminers.BasePromptingMiner ):

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument('--deployment_framework',  type=str, choices=['accelerate', 'deepspeed'], default="accelerate", help='Inference framework to use for multi-gpu inference')
        parser.add_argument( '--use_8_bit', action='store_true', default=False, help='Whether to use int8 quantization or not.' )
        parser.add_argument( '--use_4_bit',  action='store_true', default=False, help='Whether to use int4 quantization or not' )
        parser.add_argument('--llama.model_name',  type=str, default="huggyllama/llama-65b", help='Name/path of model to load')
        parser.add_argument('--llama.max_tokens', type=int, default=20, help="The maximum number of tokens to generate in the completion.")
        parser.add_argument('--llama.do_sample', type=bool, default=True, help='Description of do_sample')
        parser.add_argument('--llama.temperature', type=float, default=1.0, help='Description of temperature')
        parser.add_argument('--llama.top_p', type=float, default=0.95, help='Description of top_p')
        parser.add_argument('--llama.top_k', type=int, default=10, help='Description of top_k')
        parser.add_argument('--llama.stopping_criteria', type=str, default='stop', help='Description of stopping_criteria')

    @classmethod
    def config( cls ) -> "bittensor.Config":
        parser = argparse.ArgumentParser( description='Falcon Miner Config' )
        cls.add_args( parser )
        return bittensor.config( parser )

    def __init__( self, *args, **kwargs):
        super( LlamaMiner, self ).__init__( *args, **kwargs )
        bittensor.logging.info( 'Loading ' + str( self.config.llama.model_name ) )
        # loading the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llama.model_name)

        if self.config.deployment_framework == "deepspeed":        

            # distributed setup
            os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid warnings about parallelism in tokenizers
            self.local_rank = int(os.getenv('LOCAL_RANK', '0'))
            world_size = int(os.getenv('WORLD_SIZE', '1'))
            torch.cuda.set_device(self.local_rank)
            deepspeed.init_distributed()

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llama.model_name)

            config = AutoConfig.from_pretrained(self.config.llama.model_name, trust_remote_code=True)

            model_hidden_size = config.hidden_size

            # batch size has to be divisible by world_size, but can be bigger than world_size
            train_batch_size = 1 * world_size

            # ds_config notes
            #
            # - enable bf16 if you use Ampere or higher GPU - this will run in mixed precision and will be
            # faster.
            #
            # - for older GPUs you can enable fp16, but it'll only work for non-bf16 pretrained models - e.g.
            # all official t5 models are bf16-pretrained
            #
            # - set offload_param.device to "none" or completely remove the `offload_param` section if you don't
            # - want CPU offload
            #
            # - if using `offload_param` you can manually finetune stage3_param_persistence_threshold to control
            # - which params should remain on gpus - the larger the value the smaller the offload size
            #
            # For indepth info on Deepspeed config see
            # https://huggingface.co/docs/transformers/master/main_classes/deepspeed
            ds_config = {
                "fp16": {
                    "enabled": False,
                },
                "bf16": {
                    "enabled": False,
                },
                "zero_optimization": {
                    "stage": 3,
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

            # next line instructs transformers to partition the model directly over multiple gpus using
            # deepspeed.zero.Init when model's `from_pretrained` method is called.
            #
            # **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
            #
            # otherwise the model will first be loaded normally and only partitioned at forward time which is
            # less efficient and when there is little CPU RAM may fail
            dschf = HfDeepSpeedConfig(ds_config) # keep this object alive

            # now a model can be loaded.
            self.model = AutoModelForCausalLM.from_pretrained(self.config.llama.model_name, trust_remote_code=True)

            # initialise deepspeed ZeRO
            self.ds_engine = deepspeed.initialize(model=self.model,
                                            config_params=ds_config,
                                            model_parameters=None,
                                            optimizer=None,
                                            lr_scheduler=None)[0]
            self.ds_engine.module.eval() 
        
        else:

            if self.config.use_8_bit and self.config.use_4_bit:
                raise ValueError(
                    "You can't use 8 bit and 4 bit precision at the same time"
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.llama.model_name, 
                device_map="auto", 
                torch_dtype=torch.float16, 
                load_in_8bit=self.config.use_8_bit, 
                load_in_4bit=self.config.use_4_bit,
            )

            self.pipe = pipeline( 
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

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
        if self.config.deployment_framework == "deepspeed":
            t_generate_start = time.time()
            inputs = self.tokenizer.encode(history, return_tensors="pt").to(device=self.local_rank)
            with torch.no_grad():
                outputs = self.ds_engine.module.generate(inputs, max_length= 60)
            resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace( str( history ), "")
        else:
            resp = self.pipe( 
                history, 
                max_length=200,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id, 
            )[0]['generated_text'].split(':')[-1].replace( str( history ), "")

        # Logging input and generation if debugging is active
        bittensor.logging.debug( "Message: " + str( messages ) )
        bittensor.logging.debug( "Generation: " + str( resp ) )
        return resp

if __name__ == "__main__":  
    miner = LlamaMiner()
    with miner:
        while True:
            print ('running...', time.time() )
            time.sleep( 1 )