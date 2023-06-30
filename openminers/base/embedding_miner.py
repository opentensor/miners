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

import torch
import bittensor as bt

from abc import ABC
from typing import List, Union, Tuple

from .forward import forward
from .priority import priority
from .blacklist import blacklist
from .miner import BaseMiner


class BaseEmbeddingMiner(BaseMiner, ABC):
    def __init__(self, *args, **kwargs):
        super(BaseEmbeddingMiner, self).__init__(*args, **kwargs)

        class Synapse(bt.TextToEmbeddingSynapse):

            # Build priority function.
            def priority(_, forward_call: "bt.TextToEmbeddingForwardCall") -> float:
                return priority(self, self.priority, forward_call)
                # return 0.0

            # Build blacklist function.
            def blacklist(
                _, forward_call: "bt.TextToEmbeddingForwardCall"
            ) -> Union[Tuple[bool, str], bool]:
                return blacklist(self, self.blacklist, forward_call)

            # Build forward function.
            def forward(_, text: List[str]) -> str:
                return forward(self, self.forward, text)

            # Build backward function.
            # TODO(const): accept this.
            def backward(
                self, text: List[str], response: str, rewards: torch.FloatTensor
            ) -> str:
                pass

        # Instantiate synapse.
        self.synapse = Synapse(axon=self.axon)
