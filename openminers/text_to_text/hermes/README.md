# Usage
Specify CUDA_VISIBLE_DEVICES to select which GPUs to use. NOTE: the Hermes model requires ~50Gb VRAM, so you may want to use 2 or more GPUs. This can easily be done by prefixing CUDA_VISIBLE_DEVICES=<0,1...,N>

e.g.
```bash
CUDA_VISIBLE_DEVICES=2,3 python miners/openminers/text_to_text/hermes/miner.py
```

This will use the 2nd and 3rd GPU on your machine.