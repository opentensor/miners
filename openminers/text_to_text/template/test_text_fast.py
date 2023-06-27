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

"""
The goal of this script is to perform a test request to a hypothetical server running on localhost, at port 8092. 
The server is supposed to have a service running at /TextToCompletion/Forward, which accepts POST requests with 
data in the format specified, and returns a completed text based on the inputs. This is useful to simulate real-world 
usage and test the functionality of your services.
"""

import json
import torch
import requests
import bittensor as bt

# Enable tracing for bittensor. This provides detailed logs about the operations.
bt.trace()

# This is the hotkey for your miner. You must register this unless you set --blacklist.allow_non_registered 
# to true in the config or return False in the Synapse().blacklist() function.
hotkey = '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'

# Here we prepare the data object which will be sent as a POST request to the server. This includes
# a list of "roles" and "messages". "roles" indicate who said each message in the "messages" list. 
# "system" and "user" are example roles here. You can add as many messages as you want, in the order
# they are said. Finally, a timeout value is provided to avoid waiting indefinitely for a response.
data = {
    "roles": ["system", "user"],
    "messages": ["You are an unhelpful assistant.", "What is the capital of Texas?"],
    "timeout": 12,
}

# The post request is sent to the /TextToCompletion/Forward endpoint of the server, 
# including the hotkey and data in the request. The server is expected to take the 
# roles and messages, perform the necessary operations, and return the completed text.
req = requests.post('http://127.0.0.1:8092/TextToCompletion/Forward/?hotkey={}'.format(hotkey), json=data)

# Print the status and content of the response from the server.
print(req)
print(req.text)
