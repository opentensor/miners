import torch
import wandb
import bittensor as bt


def set_weights(
    subtensor: "bt.subtensor",
    netuid: int,
    uid: int,
    wallet: "bt.wallet",
    wandb_on=False,
) -> None:
    try:
        # --- query the chain for the most current number of peers on the network
        chain_weights = torch.zeros(subtensor.subnetwork_n(netuid=netuid))
        chain_weights[uid] = 1

        # --- Set weights.
        subtensor.set_weights(
            uids=torch.arange(0, len(chain_weights)),
            netuid=netuid,
            weights=chain_weights,
            wait_for_inclusion=False,
            wallet=wallet,
            version_key=1,
        )
        if wandb_on:
            wandb.log({"set_weights": 1})

    except Exception as e:
        if wandb_on:
            wandb.log({"set_weights": 0})
        bt.logging.error(f"Failed to set weights on chain with exception: { e }")
