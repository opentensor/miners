# Opentensor Miners
Opentensor Miners

# Installing
With ≥python3.8 installed:
```bash
python3 -m pip install -e.
```

# Running openminers
```bash
import openminers
with openminers.TemplateMiner():
    pass
```

# Running tests
```bash
pytest -s tests/
```

# Running miners
```bash
import openminers
```

# TODO
[x] Create this repo
[x] Add setup.py
[x] Add template text_to_text miner
[x] Add template text_to_text forward tests 
[ ] Add benchmarks
[ ] Add more miners
[ ] Advance blacklist function
[ ] Advance priority function
[ ] Advance telemetry through wandb