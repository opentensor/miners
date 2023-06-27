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
import torch
import requests
import bittensor as bt

# Enable tracing for bittensor. This provides detailed logs about the operations.
bt.trace()

# This is the hotkey for your miner. You must register this unless you set --blacklist.allow_non_registered 
# to true in the config or return False in the Synapse().blacklist() function.
hotkey = '<Your hotkey here>'

# This is the text for which you want to generate the embedding.
text = "(beautiful) (best quality) mdjrny-v4 style Pepe the frog enormous, surrounded by colorful sea creatures and plants, - surreal, Dreamlike, ethereal lighting, Highly detailed, Intricate, Digital painting, Artstation, Concept art, Smooth, Sharp focus, Fantasy, trending on art websites, art by magali villeneuve and jock and ashley wood and rachel lee and loish"

# Alternatively, you can also use a list of texts.
texts = ["This is a list of strings", "A second string"]

# Package the request to be parsed by a Pydantic BaseModel on the server. This object should include all 
# parameters needed for your miner to perform its task.
data = {
  "text": text,
  "timeout": 12,
}

# Send the request to the server and print the response. This request is sent to the /TextToEmbedding/Forward endpoint.
# Replace this endpoint with your miner's endpoint for different modalities.
req = requests.post('http://127.0.0.1:8092/TextToEmbedding/Forward/?hotkey={}'.format(hotkey), json=data)
print(req)
print(req.text)

# Parse the response embedding. The response should be a JSON serialized tensor.
# TODO: this should be a base64 encoded string, not List[List[float]]
x = json.loads(req.text)
emb = torch.Tensor(x)
print("emb.shape: ", emb.shape)
