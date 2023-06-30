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
import wandb
import bittensor as bt
from .set_weights import set_weights


def run(self):
    bt.logging.info(f"Starting miner with config {self.config}")

    # --- Optionally register the wallet.
    if not self.config.miner.no_register:
        bt.logging.info(
            f"Registering with wallet: {self.wallet} on netuid {self.config.netuid}"
        )
        self.subtensor.register(netuid=self.config.netuid, wallet=self.wallet)

    # --- Optionally server the axon.
    if not self.config.miner.no_serve:
        bt.logging.info(f"Serving axon: {self.axon}")
        self.subtensor.serve_axon(netuid=self.config.netuid, axon=self.axon)

    # --- Optionally start the axon.
    if not self.config.miner.no_start_axon:
        bt.logging.info(
            f"Starting axon locally on {self.axon.full_address} and serving on {self.axon.external_ip}:{self.axon.external_port}"
        )
        self.axon.start()

    # --- Run until should_exit = True.
    self.last_epoch_block = self.subtensor.get_current_block()
    bt.logging.info(f"Miner starting at block: { self.last_epoch_block }")
    while not self.should_exit:
        start_epoch = time.time()

        # --- Wait until next epoch.
        current_block = self.subtensor.get_current_block()
        while (
            current_block - self.last_epoch_block
        ) < self.config.miner.blocks_per_epoch:

            # --- Wait for next block.
            time.sleep(1)
            current_block = self.subtensor.get_current_block()

            # --- Check if we should exit.
            if self.should_exit:
                break

        # --- Update the metagraph with the latest network state.
        self.last_epoch_block = self.subtensor.get_current_block()
        self.metagraph.sync(lite=False, subtensor=self.subtensor)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # --- Log performance.
        step_log = {
            "epoch_time": time.time() - start_epoch,
            "block": self.last_epoch_block,
            "uid": self.wallet.hotkey.ss58_address,
            "stake": self.metagraph.S[self.uid].item(),
            "trust": self.metagraph.T[self.uid].item(),
            "incentive": self.metagraph.I[self.uid].item(),
            "consensus": self.metagraph.C[self.uid].item(),
            "dividends": self.metagraph.D[self.uid].item(),
        }
        bt.logging.info(str(step_log))
        if self.config.wandb.on:
            wandb.log(step_log)

        # --- Set weights.
        if not self.config.miner.no_set_weights:
            set_weights(
                self.subtensor,
                self.config.netuid,
                self.uid,
                self.wallet,
                self.config.wandb.on,
            )
