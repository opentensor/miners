# Opentensor Miners
Opentensor Miners

# Installing
With ≥python3.8 installed:
```bash
python3 -m pip install -e .
```

# Running openminers
```python
import openminers
with openminers.TemplateMiner():
    pass

# or
miner = openminers.TemplateMiner()
miner.run()

# or
miner = openminers.TemplateMiner()
miner.run_in_background_thread()
...
miner.stop_background_thread()
```

# Running Benchmarks
```bash
# Run 10 requests through the template miner
python3 benchmarks/base.py template 10

# Run 10 requests through the template miner with wandb logging
python3 benchmarks/base.py template 1000 --wandb.on 

# Run 100 requests through the openai miner with api key
python3 benchmarks/base.py openai 100 --openai.api_key xxx...xx
```

# TODO
- [x] Create this repo
- [x] Add setup.py
- [x] Add template text_to_text miner
- [x] Add template text_to_text forward tests 
- [x] Add benchmarks
- [x] Add Wandb to miner
- [x] Add more miners
- [x] Advance blacklist function
- [x] Advance priority function
- [x] Advance telemetry through wandb
- [x] Add benchmarks to miners
- [ ] Bechmark all miners
- [ ] Allow in code api key and argument pass through 
- [ ] Move set weights into its own file
- [ ] Maintain a history of requests per hotkey
- [ ] Priority based on rate limit
- [ ] Blacklist based on rate limit
- [ ] Benchmark longer and shorter request sizes
