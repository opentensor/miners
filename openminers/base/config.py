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
import openminers
import bittensor as bt


def add_args(cls, parser: argparse.ArgumentParser):

    # Add args for the super.
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)
    parser.add_argument(
        "--miner.root",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="~/.bittensor/miners/",
    )
    parser.add_argument(
        "--miner.name",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="text_to_text",
    )

    # Run config.
    parser.add_argument(
        "--miner.blocks_per_epoch",
        type=str,
        help="Blocks until the miner sets weights on chain",
        default=100,
    )

    # Blacklist.
    parser.add_argument(
        "--miner.blacklist.blacklist",
        type=str,
        required=False,
        nargs="*",
        help="Blacklist certain hotkeys",
        default=[],
    )
    parser.add_argument(
        "--miner.blacklist.whitelist",
        type=str,
        required=False,
        nargs="*",
        help="Whitelist certain hotkeys",
        default=[],
    )
    parser.add_argument(
        "--miner.blacklist.force_validator_permit",
        action="store_true",
        help="Only allow requests from validators",
        default=False,
    )
    parser.add_argument(
        "--miner.blacklist.allow_non_registered",
        action="store_true",
        help="If True, the miner will allow non-registered hotkeys to mine.",
        default=False,
    )
    parser.add_argument(
        "--miner.blacklist.minimum_stake_requirement",
        type=float,
        help="Minimum stake requirement",
        default=0.0,
    )

    # Priority.
    parser.add_argument(
        "--miner.priority.default",
        type=float,
        help="Default priority of non-registered requests",
        default=0.0,
    )
    parser.add_argument(
        "--miner.priority.use_s", type=float, help="A multiplier", default=0.0
    )

    # Switches.
    parser.add_argument(
        "--miner.no_set_weights",
        action="store_true",
        help="If True, the miner does not set weights.",
        default=False,
    )
    parser.add_argument(
        "--miner.no_serve",
        action="store_true",
        help="If True, the miner doesnt serve the axon.",
        default=False,
    )
    parser.add_argument(
        "--miner.no_start_axon",
        action="store_true",
        help="If True, the miner doesnt start the axon.",
        default=False,
    )
    parser.add_argument(
        "--miner.no_register",
        action="store_true",
        help="If True, the miner doesnt register its wallet.",
        default=False,
    )

    # Mocks.
    parser.add_argument(
        "--miner.mock_subtensor",
        action="store_true",
        help="If True, the miner will allow non-registered hotkeys to mine.",
        default=False,
    )

    # Wandb
    parser.add_argument(
        "--wandb.on", action="store_true", help="Turn on wandb.", default=False
    )
    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where youre sending the new run.",
        default=None,
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="An entity is a username or team name where youre sending runs.",
        default=None,
    )

    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)


def config(cls) -> "bt.Config":
    parser = argparse.ArgumentParser()
    add_args(cls, parser)
    return bt.config(parser)


def help(cls):
    parser = argparse.ArgumentParser()
    add_args(cls, parser)
    print(cls.__new__.__doc__)
    parser.print_help()


def check_config(cls, config: "bt.Config"):
    bt.axon.check_config(config)
    bt.wallet.check_config(config)
    bt.logging.check_config(config)
    bt.subtensor.check_config(config)
    full_path = os.path.expanduser(
        "{}/{}/{}/{}".format(
            config.logging.logging_dir,
            config.wallet.get("name", bt.defaults.wallet.name),
            config.wallet.get("hotkey", bt.defaults.wallet.hotkey),
            config.miner.name,
        )
    )
    config.miner.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.miner.full_path):
        os.makedirs(config.miner.full_path)
