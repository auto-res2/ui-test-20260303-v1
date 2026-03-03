"""Main orchestrator for EA-CoT experiment."""

import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from src.inference import run_inference


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running a single experiment.
    Orchestrates inference based on the method type.
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: ConfigAttributeError: Key 'method' is not in struct (line 18)
    # [CAUSE]: Run config is loaded under cfg.run namespace, not merged at top level
    # [FIX]: Access method via cfg.run.method instead of cfg.method
    #
    # [OLD CODE]:
    # print(f"Method: {cfg.method.type}")
    # cfg.dataset.num_tuning = ...
    #
    # [NEW CODE]:
    print("=" * 80)
    print(f"Running: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method.type}")
    print("=" * 80)

    # Validate required fields
    if not hasattr(cfg, "run") or not hasattr(cfg.run, "run_id"):
        raise ValueError("Config must have run.run_id field")

    # Override settings based on mode
    if cfg.mode == "sanity_check":
        # For sanity check: use fewer examples, online wandb
        cfg.run.dataset.num_tuning = min(10, cfg.run.dataset.num_tuning)
        cfg.run.dataset.num_eval = min(10, cfg.run.dataset.num_eval)
        cfg.wandb.mode = "online"
    elif cfg.mode == "main":
        # For main runs: ensure online wandb
        cfg.wandb.mode = "online"

    print(f"\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Run inference (this is an inference-only task)
    try:
        run_inference(cfg)
        print(f"\n{'=' * 80}")
        print(f"Successfully completed: {cfg.run.run_id}")
        print(f"{'=' * 80}")
    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"ERROR in {cfg.run.run_id}: {str(e)}")
        print(f"{'=' * 80}")
        raise


if __name__ == "__main__":
    main()
