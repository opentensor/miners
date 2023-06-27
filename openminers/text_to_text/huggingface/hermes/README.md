
## Hermes Miner
Nous Hermes completion miner for bittensor's prompting network. 

# Example Usage
```
python3 openminers/text_to_text/huggingface/hermes/miner.py --miner.model_name <repo_id/repo_name>
```

# Full Usage
```
usage: miner.py [-h] [--netuid NETUID] [--miner.root MINER.ROOT] [--miner.name MINER.NAME] [--miner.blocks_per_epoch MINER.BLOCKS_PER_EPOCH]
                [--miner.blacklist.blacklist [MINER.BLACKLIST.BLACKLIST [MINER.BLACKLIST.BLACKLIST ...]]] [--miner.blacklist.whitelist [MINER.BLACKLIST.WHITELIST [MINER.BLACKLIST.WHITELIST ...]]]
                [--miner.blacklist.force_validator_permit] [--miner.blacklist.allow_non_registered] [--miner.blacklist.minimum_stake_requirement MINER.BLACKLIST.MINIMUM_STAKE_REQUIREMENT]
                [--miner.priority.default MINER.PRIORITY.DEFAULT] [--miner.priority.use_s MINER.PRIORITY.USE_S] [--miner.no_set_weights] [--miner.no_serve] [--miner.no_start_axon] [--miner.no_register]
                [--miner.mock_subtensor] [--wandb.on] [--wandb.project_name WANDB.PROJECT_NAME] [--wandb.entity WANDB.ENTITY] [--wallet.name WALLET.NAME] [--wallet.hotkey WALLET.HOTKEY] [--wallet.path WALLET.PATH]
                [--wallet._mock] [--wallet.reregister WALLET.REREGISTER] [--axon.priority.max_workers AXON.PRIORITY.MAX_WORKERS] [--axon.priority.maxsize AXON.PRIORITY.MAXSIZE] [--axon.port AXON.PORT] [--axon.ip AXON.IP]
                [--axon.external_port AXON.EXTERNAL_PORT] [--axon.external_ip AXON.EXTERNAL_IP] [--axon.max_workers AXON.MAX_WORKERS] [--axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS]
                [--subtensor.network SUBTENSOR.NETWORK] [--subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT] [--subtensor._mock] [--subtensor.register.num_processes SUBTENSOR.REGISTER.NUM_PROCESSES]
                [--subtensor.register.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL] [--subtensor.register.no_output_in_place] [--subtensor.register.verbose] [--subtensor.register.cuda.use_cuda]
                [--subtensor.register.cuda.no_cuda] [--subtensor.register.cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...]] [--subtensor.register.cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB]
                [--logging.debug] [--logging.trace] [--logging.record_log] [--logging.logging_dir LOGGING.LOGGING_DIR] [--config CONFIG] [--strict]

optional arguments:
  -h, --help            show this help message and exit
  --netuid NETUID       Subnet netuid
  --miner.root MINER.ROOT
                        Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name
  --miner.name MINER.NAME
                        Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name
  --miner.blocks_per_epoch MINER.BLOCKS_PER_EPOCH
                        Blocks until the miner sets weights on chain
  --miner.blacklist.blacklist [MINER.BLACKLIST.BLACKLIST [MINER.BLACKLIST.BLACKLIST ...]]
                        Blacklist certain hotkeys
  --miner.blacklist.whitelist [MINER.BLACKLIST.WHITELIST [MINER.BLACKLIST.WHITELIST ...]]
                        Whitelist certain hotkeys
  --miner.blacklist.force_validator_permit
                        Only allow requests from validators
  --miner.blacklist.allow_non_registered
                        If True, the miner will allow non-registered hotkeys to mine.
  --miner.blacklist.minimum_stake_requirement MINER.BLACKLIST.MINIMUM_STAKE_REQUIREMENT
                        Minimum stake requirement
  --miner.priority.default MINER.PRIORITY.DEFAULT
                        Default priority of non-registered requests
  --miner.priority.use_s MINER.PRIORITY.USE_S
                        A multiplier
  --miner.no_set_weights
                        If True, the miner does not set weights.
  --miner.no_serve      If True, the miner doesnt serve the axon.
  --miner.no_start_axon
                        If True, the miner doesnt start the axon.
  --miner.no_register   If True, the miner doesnt register its wallet.
  --miner.mock_subtensor
                        If True, the miner will allow non-registered hotkeys to mine.
  --wandb.on            Turn on wandb.
  --wandb.project_name WANDB.PROJECT_NAME
                        The name of the project where youre sending the new run.
  --wandb.entity WANDB.ENTITY
                        An entity is a username or team name where youre sending runs.
  --wallet.name WALLET.NAME
                        The name of the wallet to unlock for running bittensor (name mock is reserved for mocking this wallet)
  --wallet.hotkey WALLET.HOTKEY
                        The name of wallet's hotkey.
  --wallet.path WALLET.PATH
                        The path to your bittensor wallets
  --wallet._mock        To turn on wallet mocking for testing purposes.
  --wallet.reregister WALLET.REREGISTER
                        Whether to reregister the wallet if it is not already registered.
  --axon.priority.max_workers AXON.PRIORITY.MAX_WORKERS
                        maximum number of threads in thread pool
  --axon.priority.maxsize AXON.PRIORITY.MAXSIZE
                        maximum size of tasks in priority queue
  --axon.port AXON.PORT
                        The local port this axon endpoint is bound to. i.e. 8091
  --axon.ip AXON.IP     The local ip this axon binds to. ie. [::]
  --axon.external_port AXON.EXTERNAL_PORT
                        The public port this axon broadcasts to the network. i.e. 8091
  --axon.external_ip AXON.EXTERNAL_IP
                        The external ip this axon broadcasts to the network to. ie. [::]
  --axon.max_workers AXON.MAX_WORKERS
                        The maximum number connection handler threads working simultaneously on this endpoint. The grpc server distributes new worker threads to service requests up to this number.
  --axon.maximum_concurrent_rpcs AXON.MAXIMUM_CONCURRENT_RPCS
                        Maximum number of allowed active connections
  --subtensor.network SUBTENSOR.NETWORK
                        The subtensor network flag. The likely choices are: -- finney (main network) -- local (local running network) -- mock (creates a mock connection (for testing)) If this option is set it overloads
                        subtensor.chain_endpoint with an entry point node from that network.
  --subtensor.chain_endpoint SUBTENSOR.CHAIN_ENDPOINT
                        The subtensor endpoint flag. If set, overrides the --network flag.
  --subtensor._mock     To turn on subtensor mocking for testing purposes.
  --subtensor.register.num_processes SUBTENSOR.REGISTER.NUM_PROCESSES, -n SUBTENSOR.REGISTER.NUM_PROCESSES
                        Number of processors to use for registration
  --subtensor.register.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL, --subtensor.register.cuda.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL, --cuda.update_interval SUBTENSOR.REGISTER.UPDATE_INTERVAL, -u SUBTENSOR.REGISTER.UPDATE_INTERVAL
                        The number of nonces to process before checking for next block during registration
  --subtensor.register.no_output_in_place, --no_output_in_place
                        Whether to not ouput the registration statistics in-place. Set flag to disable output in-place.
  --subtensor.register.verbose
                        Whether to ouput the registration statistics verbosely.
  --subtensor.register.cuda.use_cuda, --cuda, --cuda.use_cuda
                        Set flag to use CUDA to register.
  --subtensor.register.cuda.no_cuda, --no_cuda, --cuda.no_cuda
                        Set flag to not use CUDA for registration
  --subtensor.register.cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...], --cuda.dev_id SUBTENSOR.REGISTER.CUDA.DEV_ID [SUBTENSOR.REGISTER.CUDA.DEV_ID ...]
                        Set the CUDA device id(s). Goes by the order of speed. (i.e. 0 is the fastest).
  --subtensor.register.cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB, --cuda.TPB SUBTENSOR.REGISTER.CUDA.TPB
                        Set the number of Threads Per Block for CUDA.
  --logging.debug       Turn on bittensor debugging information
  --logging.trace       Turn on bittensor trace level information
  --logging.record_log  Turns on logging to file.
  --logging.logging_dir LOGGING.LOGGING_DIR
                        Logging default root directory.
  --config CONFIG       If set, defaults are overridden by passed file.
  --strict              If flagged, config will check that only exact arguemnts have been set.
```