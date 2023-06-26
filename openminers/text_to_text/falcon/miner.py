# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# Copyright © 2021 Yuma Rao

# The MIT License (MIT)
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
import deepspeed
import os

from typing import List, Dict
from transformers import AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig

class StopOnTokens( StoppingCriteria ):

    def __init__( self, stop_token_ids: List[int] ):
        self.stop_token_ids = stop_token_ids

    def __call__( self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs ) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class FalconMiner( openminers.BasePromptingMiner ):

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--deployment_framework',  type=str, choices=['accelerate', 'deepspeed'], default="accelerate", help='Inference framework to use for multi-gpu inference')
        parser.add_argument( '--falcon.model_name', type=str, default="tiiuae/falcon-7b-instruct", help='Name/path of model to load' )
        parser.add_argument( '--falcon.device', type=int, help='Device to load model (integer GPU slot)', default=0 )
        parser.add_argument( '--falcon.device_map', type=str, help='Device map for model', default="auto" )
        parser.add_argument( '--falcon.max_length', type=int, help='Max tokens for model output.', default=100 )
        parser.add_argument( '--falcon.temperature', type=float, help='Sampling temperature of model', default=0.5 )
        parser.add_argument( '--falcon.top_k', type=int, help='Top k sampling of model', default=1 )
        parser.add_argument( '--falcon.do_sample', action='store_true', default=False, help='Whether to use sampling or not (if not, uses greedy decoding).' )
        parser.add_argument( '--falcon.do_prompt_injection', action='store_true', default=False, help='Whether to use a custom "system" prompt instead of the one sent by bittensor.' )
        parser.add_argument( '--falcon.system_prompt', type=str, help='What prompt to replace the system prompt with', default= "A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions. " )
        parser.add_argument( '--falcon.num_return_sequences', type=int, help='Number of sequences to return', default=1 )
        parser.add_argument( '--falcon.repetition_penalty', type=float, help='Repetition penalty for model', default=1.9 )

    @classmethod
    def config( cls ) -> "bittensor.Config":
        parser = argparse.ArgumentParser( description='Falcon Miner Config' )
        cls.add_args( parser )
        return bittensor.config( parser )

    def __init__( self, *args, **kwargs):
        super( FalconMiner, self ).__init__( *args, **kwargs )

        bittensor.logging.info( 'Loading ' + str( self.config.falcon.model_name ) )
        self.tokenizer = AutoTokenizer.from_pretrained( self.config.falcon.model_name )
        self.stop_token_ids = self.tokenizer.convert_tokens_to_ids( ["</s>","<|endoftext|>"] )
        self.stop = StopOnTokens( self.stop_token_ids )

        if self.config.deployment_framework == "deepspeed":
            # distributed setup
            os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid warnings about parallelism in tokenizers
            self.local_rank = int(os.getenv('LOCAL_RANK', '0'))
            world_size = int(os.getenv('WORLD_SIZE', '1'))
            torch.cuda.set_device(self.local_rank)
            deepspeed.init_distributed()

            self.model = AutoModelForCausalLM.from_pretrained(self.config.falcon.model_name, trust_remote_code=True)
            model_hidden_size = self.model.config.hidden_size

            # batch size has to be divisible by world_size, but can be bigger than world_size
            train_batch_size = 1 * world_size

            # ds_config variables
            ds_config = {
                "fp16": {
                    "enabled": False,
                },
                "bf16": {
                    "enabled": True,
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
            dschf = HfDeepSpeedConfig(ds_config)

            # initialise deepspeed ZeRO
            self.ds_engine = deepspeed.initialize(model=self.model,
                                            config_params=ds_config,
                                            model_parameters=None,
                                            optimizer=None,
                                            lr_scheduler=None)[0]
            self.ds_engine.module.eval() 


        else:
            kwargs = dict(
                model=self.config.falcon.model_name,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                temperature=self.config.falcon.temperature,
                do_sample=self.config.falcon.do_sample,
                device_map=self.config.falcon.device_map
            )

            if self.config.falcon.device_map is not None:
                kwargs['device_map'] = self.config.falcon.device_map
            else:
                kwargs['device'] = self.config.falcon.device

            self.model = pipeline( "text-generation",  **kwargs )
            bittensor.logging.info( 'Model loaded!' )

            # if self.config.falcon.device != "cpu" and self.config.falcon.device_map is not None:
            #     self.model = self.model.to( self.config.falcon.device )

    def _process_history( self, history: List[str] ) -> str:
        processed_history = ''
        if self.config.falcon.do_prompt_injection:
            processed_history += self.config.falcon.system_prompt
        for message in history:
            if message['role'] == 'system':
                if not self.config.falcon.do_prompt_injection or message != history[0]:
                    processed_history += '' + message['content'].strip() + ' '
            if message['role'] == 'assistant':
                processed_history += 'Assistant:' + message['content'].strip() + '</s>'
            if message['role'] == 'user':
                processed_history += 'User: ' + message['content'].strip() + ' '
        return processed_history

    def forward( self, messages: List[Dict[str, str]] ) -> str:
        history = self._process_history( messages )
        prompt = history + "ASSISTANT:"
        
        if self.config.deployment_framework == "deepspeed":
            t_generate_start = time.time()
            inputs = self.tokenizer.encode(history, return_tensors="pt").to(device=self.local_rank)
            with torch.no_grad():
                outputs = self.ds_engine.module.generate(inputs, max_length= 60)
            generation = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace( str( history ), "")
            print(generation)
            t_generate_span = time.time() - t_generate_start
            print(t_generate_span)
        
        else:
            generation = self.model(
                prompt,
                max_length=self.config.falcon.max_length,
                do_sample=self.config.falcon.do_sample,
                top_k=self.config.falcon.top_k,
                num_return_sequences=self.config.falcon.num_return_sequences,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=self.config.falcon.repetition_penalty,
                stopping_criteria=StoppingCriteriaList( [self.stop] ),
            )[0]['generated_text'].split(':')[-1].replace( str( history ), "")

        # Logging input and generation if debugging is active
        bittensor.logging.debug( "Message: " + str( messages ) )
        bittensor.logging.debug( "Generation: " + str( generation ) )
        return generation

if __name__ == "__main__":
    # FalconMiner().run()
    prompt = """you are a chatbot that can come up with unique questions about many things."""
    message = "ask me a random question about anything"
    roles = ['system', 'user']
    messages = [{"role":"system", "content":"you are a chatbot that can come up with unique questions about many things."}, {"role":"user", "content":"ask me a random question about anything"}]
    # messages = [ prompt, message ]
    print(FalconMiner().forward(messages))
    # with FalconMiner():
    #     while True:
    #         print ('running...', time.time() )
    #         time.sleep( 1 )
