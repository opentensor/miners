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

import time
import torch
import argparse
import bittensor
import openminers
from typing import Union, Tuple, List

from InstructorEmbedding import INSTRUCTOR


class InstructorEmbeddingMiner(openminers.BaseEmbeddingMiner):
    @classmethod
    def check_config(cls, config: "bittensor.Config"):
        pass

    @classmethod
    def config(cls) -> "bittensor.Config":
        parser = argparse.ArgumentParser(
            description="Instructor Embedding Miner Configs"
        )
        cls.add_args(parser)
        return bittensor.config(parser)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--instructor.model_name",
            type=str,
            help="Name of the Instructor Embedding model to use.",
            default="hkunlp/instructor-xl",
        )
        parser.add_argument(
            "--instructor.device", type=str, help="Device to load model", default="cuda"
        )

    def __init__(self, *args, **kwargs):
        super(InstructorEmbeddingMiner, self).__init__(*args, **kwargs)
        self.model = INSTRUCTOR(self.config.instructor.model_name)
        self.device = torch.device(self.config.instructor.device)
        self.model.to(self.device)

    def blacklist(
        self, forward_call: "bittensor.TextToEmbeddingForwardCall"
    ) -> Union[Tuple[bool, str], bool]:
        return False

    def priority(self, forward_call: "bittensor.TextToEmbeddingForwardCall") -> float:
        return 0.0

    def forward(self, text: List[str]) -> torch.FloatTensor:
        embedded_text = self.model.encode(text)
        embedding_tensor = torch.tensor(embedded_text)
        return embedding_tensor


if __name__ == "__main__":
    with InstructorEmbeddingMiner():
        while True:
            time.sleep(1)
