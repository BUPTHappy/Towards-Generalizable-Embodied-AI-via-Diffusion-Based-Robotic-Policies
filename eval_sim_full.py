import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
import numpy as np
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import random
from omegaconf import open_dict
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--num_sampling_steps", default="100", help="Number of diffusion sampling steps (act_diff_testing_steps)")
@click.option("--n_test", default=200, help="Number of test episodes for full dataset evaluation")
def main(checkpoint, output_dir, device, num_sampling_steps, n_test):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    # set seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir
        
        # Set diffusion sampling steps
        if "autoregressive_model_params" in cfg.model.policy:
            original_steps = cfg.model.policy.autoregressive_model_params.num_sampling_steps
            cfg.model.policy.autoregressive_model_params.num_sampling_steps = num_sampling_steps
            print(f"✓ Changed num_sampling_steps from {original_steps} to {num_sampling_steps}")
        
        # Enable action prediction to use diffusion sampling
        if "action_model_params" in cfg.model.policy:
            cfg.model.policy.action_model_params.predict_action = True
            print("✓ Enabled predict_action to use diffusion sampling")
        
        # Set full dataset evaluation
        if "env_runner" in cfg.task:
            original_n_test = cfg.task.env_runner.n_test
            cfg.task.env_runner.n_test = n_test
            print(f"✓ Changed n_test from {original_n_test} to {n_test} for full dataset evaluation")
        
    # configure workspace
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace

    print("Loaded checkpoint from %s" % checkpoint)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    
    # get policy from workspace
    policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    env_runners = load_env_runner(cfg, output_dir)

    if "libero" in cfg.task.name:
        step_log = {}
        for env_runner in env_runners:
            runner_log = env_runner.run(policy)
            step_log.update(runner_log)
            print(step_log)

        assert "test_mean_score" not in step_log
        all_test_mean_score = {
            k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
        }
        step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))

        runner_log = step_log
    else:
        env_runner = env_runners
        runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    for k, v in json_log.items():
        print(k, v)

    out_path = os.path.join(output_dir, f'eval_log_full_{checkpoint.split("/")[-1]}_steps_{num_sampling_steps}.json')
    print("Saving log to %s" % out_path)
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
