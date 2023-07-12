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
import re
import argparse
import openminers
import wandb
import torch
import numpy as np
import bittensor as bt

from typing import List, Dict, Optional
import random

class ContemplateMiner(openminers.BasePromptingMiner):
    
    choices = dict(
        opening_phrase = ['Hey, what\'s up!','...','Interesting question.','You asked the right AI!','Message received!','Yes, that\'s a good question','I see your point','I\'m happy to help!','Greetings, glad you asked.','Well now then..','I\'ve been waiting for this moment.','As an AI I can help you with that request.','Dis my moment!'],
        answer_word = ['address','fulfill','satisfy','complete','serve','respond to','correctly answer','appropriately answer','nail','unpack','unravel'],
        query_word = ['query','request','context','question'],        
        importance_conf = ['surely','quite','possibly','probably','absolutely','positively','terribly','arguably','laudibly'],
        importance_word = ['important','imperative','useful','suitable','sensible','apt','intelligent','astute','valuable','wise'],
        consider_conf = ['deeply','fully','first','primarily','tentatively','pause and','stop and','simply','carefully'],
        consider_word = ['consider','contemplate','examine','medidate on','explore','understand','digest','ponder','ruminate over','evaluate','assess','be cognizant of'],
        meaning_conf = ['intended','historical','contextual','subjective','objective','scientific','spiritual'],
        meaning_word = ['meaning','implications','value','essence','insight','utility','purpose','viewpoints','perspectives'],
    )
    
    
    template = '{opening_phrase}\nIn order to {answer_word} that {query_word} it\'s {importance_conf} {importance_word} to {consider_conf} {consider_word} the {meaning_conf} {meaning_word} of the {query_word}.'    
    
    learning_rate = 0.1
    
    @property
    def weights(self):
        if not hasattr(self, '_weights'):
            self._build_weights()
        return self._weights     
        
    @property 
    def pattern(self):
        if not hasattr(self, '_pattern'):
            self._build_pattern()
        return self._pattern
    
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--neuron.learning_rate",
            type=float,
            help="Reponse to create",
            default=0.1
        )
        parser.add_argument(
            "--neuron.max_tokens",
            type=int,
            help="Number of tokens to generate.",
            default=256,
        )
        parser.add_argument(
            "--cohere.temperature",
            type=float,
            help="Temperature of generation.",
            default=0.75,
        )
        

    def __init__(self, *args, **kwargs):
        super(ContemplateMiner, self).__init__(*args, **kwargs)
        self.learning_rate = self.config.neuron.learning_rate

    @staticmethod
    def _process_history(history: List[Dict[str, str]]) -> str:
        processed_history = ""
        for message in history:
            if message["role"] == "system":
                processed_history += "system: " + message["content"] + "\n"
            if message["role"] == "assistant":
                processed_history += "assistant: " + message["content"] + "\n"
            if message["role"] == "user":
                processed_history += "user: " + message["content"] + "\n"
        return processed_history

    def _build_weights(self):
        self._weights = {key: np.ones(len(words))/len(words) for key, words in self.choices.items()}       
            
    def _build_pattern(self):
        
        # create regex pattern to detect choices from completion     
        keywords = re.findall(r"{(.*?)}", self.template)
        pattern = re.escape(self.template)
        seen = set()
        for i, key in enumerate(keywords):
            if key not in seen:
                name = key
                seen.add(key)
            else:
                name = f'key{i}'    
            wordmatch = ('('+'|'.join(self.choices[key])+')').replace('.','\.')
            pattern = pattern.replace("\\{"+key+"\\}", f"(?P<{name}>{wordmatch})",1)    
        self._pattern = re.compile(pattern)
        
    def model(self, history):
        
        chosen_words = {key: np.random.choice(words,p=self.weights[key]) for key, words in self.choices.items()}
        return self.template.format(**chosen_words)
        
    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self._process_history(messages)
        
        return self.model(history)
    
    def backward(self, messages: List[Dict[str, str]], response: str, rewards: "torch.FloatTensor") -> float:

        # Use the pattern to extract the values            
        chosen_words = self.pattern.match(response).groupdict()
        # print(f'chosen_words: {chosen_words}')
        choice_indices = {key: words.index(chosen_words[key]) for key, words in self.choices.items()}
            
        # increment the weights of those choices by the rewards for the completion
        for key, index in choice_indices.items():
            self.weights[key][index] += self.learning_rate * rewards
            self.weights[key]/=self.weights[key].sum()        

if __name__ == "__main__":
    with ContemplateMiner():
        while True:
            print("running...", time.time())
            time.sleep(1)
