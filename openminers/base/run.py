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
import datetime
import openminers
import bittensor as bt

def run( self ):
    bt.logging.info( f"Starting miner with config {self.config}" )

    # --- Start the miner.
    bt.logging.info( f"Registering with wallet: {self.wallet} on netuid {self.config.netuid}" )
    self.subtensor.register( netuid = self.config.netuid, wallet = self.wallet )

    bt.logging.info( f"Serving axon: {self.axon}" )
    self.subtensor.serve_axon( netuid = self.config.netuid, axon = self.axon )

    bt.logging.info( f"Starting axon locally on {self.axon.full_address} and serving on {self.axon.external_ip}:{self.axon.external_port}" )
    self.axon.start()

    # --- Run Forever.
    last_update = self.subtensor.get_current_block()
    bt.logging.info( f"Miner starting at block: { last_update }" )
    while not self.should_exit:

        # --- Wait until next epoch.
        current_block = self.subtensor.get_current_block()
        while (current_block - last_update) < self.config.miner.blocks_per_epoch:

            # --- Wait for next block.
            time.sleep( 0.1 ) #bittensor.__blocktime__
            current_block = self.subtensor.get_current_block()

            # --- Check if we should exit.
            if self.should_exit: break

        last_update = self.subtensor.get_current_block()

        # # --- Update the metagraph with the latest network state.
        # self.metagraph.sync( lite = True )
        # uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )

        # # --- Log performance.
        # print(
        #     f"[white not bold]{datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
        #     f"{f'UID [bright_cyan]{uid}[/bright_cyan]'.center(16 + len('[bright_cyan][/bright_cyan]'))} | "
        #     f'[dim white not bold] [green]{str(self.metagraph.S[uid].item()):.4}[/green] Stake [/dim white not bold]'
        #     f'[dim white not bold]| [yellow]{str(self.metagraph.trust[uid].item()) :.3}[/yellow] Trust [/dim white not bold]'
        #     f'[dim white not bold]| [green]{str(self.metagraph.incentive[uid].item()):.3}[/green] Incentive [/dim white not bold]')

        # # --- Set weights.
        # if not self.config.miner.no_set_weights:
        #     try:
        #         # --- query the chain for the most current number of peers on the network
        #         chain_weights = torch.zeros( self.subtensor.subnetwork_n( netuid = self.config.netuid ))
        #         chain_weights[uid] = 1
        #         did_set = self.subtensor.set_weights(
        #             uids=torch.arange(0, len(chain_weights)),
        #             netuid=self.config.netuid,
        #             weights=chain_weights,
        #             wait_for_inclusion=False,
        #             wallet=self.wallet,
        #             version_key=1
        #         )
        #     except:
        #         pass