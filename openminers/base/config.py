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

import os
import argparse
import bittensor as bt

def add_args( cls, parser: argparse.ArgumentParser ):
    # Call add args to on sub class.
    cls.add_args( parser )

    # Add args for the super.
    parser.add_argument( '--netuid', type = int, help = 'Subnet netuid', default = 1 )
    parser.add_argument( '--miner.root', type = str, help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default = '~/.bittensor/miners/' )
    parser.add_argument( '--miner.name', type = str, help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default = 'text_to_text')
    parser.add_argument( '--miner.blocks_per_epoch', type = str, help = 'Blocks until the miner sets weights on chain', default = 100 )
    parser.add_argument( '--miner.no_set_weights', action = 'store_true', help = 'If True, the model does not set weights.', default = False )
    parser.add_argument( '--miner.blacklist.hotkeys', type = str, required = False, nargs = '*', action = 'store', help = 'To blacklist certain hotkeys', default = [] )
    parser.add_argument( '--miner.blacklist.allow_non_registered', action = 'store_true', help = 'If True, the miner will allow non-registered hotkeys to mine.', default = False )
    parser.add_argument( '--miner.blacklist.default_stake', type = float, help = 'Set default stake for miners.', default = 0.0)
    parser.add_argument( '--miner.default_priority', type = float, help = 'Set default priority for miners.', default = 0.0 )
    parser.add_argument( '--miner.mock_subtensor', action = 'store_true', help = 'If True, the miner will allow non-registered hotkeys to mine.', default = True)

    bt.wallet.add_args( parser )
    bt.axon.add_args( parser )
    bt.subtensor.add_args( parser )
    bt.logging.add_args( parser )

def config( cls ) -> "bt.Config":
    parser = argparse.ArgumentParser()
    add_args( cls, parser )
    return bt.config( parser )

def help( cls ):
    parser = argparse.ArgumentParser()
    add_args( cls, parser )
    print( cls.__new__.__doc__ )
    parser.print_help()

def check_config( cls, config: 'bt.Config' ):
    bt.axon.check_config( config )
    bt.wallet.check_config( config )
    bt.logging.check_config( config )
    bt.subtensor.check_config( config )
    full_path = os.path.expanduser(
        '{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.get('name', bt.defaults.wallet.name),
                                config.wallet.get('hotkey', bt.defaults.wallet.hotkey), config.miner.name ) )
    config.miner.full_path = os.path.expanduser( full_path )
    if not os.path.exists( config.miner.full_path ):
        os.makedirs( config.miner.full_path )
