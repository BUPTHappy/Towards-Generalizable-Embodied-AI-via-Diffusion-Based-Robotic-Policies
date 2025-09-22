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
@click.option("-d", "--device", default="cuda:4")
@click.option("--enable_grad_checkpointing", is_flag=True, default=True, help="Enable gradient checkpointing to save memory")
@click.option("--reduce_batch_size", is_flag=True, default=True, help="Reduce effective batch size for memory")
@click.option("--use_mixed_precision", is_flag=True, default=True, help="Use mixed precision to save memory")
@click.option("--vae_chunk_size", default=2, help="Number of frames to process at once in VAE")
@click.option("--vae_batch_chunk_size", default=4, help="Number of batches to process at once in VAE")
@click.option("--act_diff_testing_steps", default="100", help="Number of diffusion sampling steps")
def main(checkpoint, output_dir, device, enable_grad_checkpointing, reduce_batch_size, use_mixed_precision, vae_chunk_size, vae_batch_chunk_size, act_diff_testing_steps):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set the specific GPU device
        torch.cuda.set_device(device)
        print(f"✓ Using GPU device: {device}")

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
            original_steps = cfg.model.policy.autoregressive_model_params.act_diff_testing_steps
            cfg.model.policy.autoregressive_model_params.num_sampling_steps = act_diff_testing_steps
            print(f"✓ Changed num_sampling_steps from {original_steps} to {act_diff_testing_steps}")
        
        # Enable action prediction to use diffusion sampling
        if "action_model_params" in cfg.model.policy:
            cfg.model.policy.action_model_params.predict_action = True
            print("✓ Enabled predict_action to use diffusion sampling")
        
        # Memory optimizations
        if enable_grad_checkpointing:
            cfg.model.policy.autoregressive_model_params.grad_checkpointing = True
            print("✓ Enabled gradient checkpointing")
            
        if reduce_batch_size:
            # Reduce sequence length for evaluation
            if hasattr(cfg.task.env_runner, 'n_obs_steps'):
                cfg.task.env_runner.n_obs_steps = min(cfg.task.env_runner.n_obs_steps, 8)
                print(f"✓ Reduced n_obs_steps to {cfg.task.env_runner.n_obs_steps}")
            
            # Reduce action steps
            if hasattr(cfg.task.env_runner, 'n_action_steps'):
                cfg.task.env_runner.n_action_steps = min(cfg.task.env_runner.n_action_steps, 4)
                print(f"✓ Reduced n_action_steps to {cfg.task.env_runner.n_action_steps}")
        
    # configure workspace
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace

    print("Loaded checkpoint from %s" % checkpoint)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None, strict=False)
    
    
    # get policy from workspace
    policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    # Enable memory optimizations for the model
    if enable_grad_checkpointing and hasattr(policy, 'model'):
        if hasattr(policy.model, 'enable_gradient_checkpointing'):
            policy.model.enable_gradient_checkpointing()
            print("✓ Enabled gradient checkpointing on model")
    
    # Enable VAE checkpointing and chunked processing
    if enable_grad_checkpointing:
        policy.use_vae_checkpointing = True
        policy.vae_chunk_size = vae_chunk_size
        policy.vae_batch_chunk_size = vae_batch_chunk_size
        print(f"✓ Enabled VAE chunked processing (chunk_size={vae_chunk_size}, batch_chunk_size={vae_batch_chunk_size})")
    
    # Use mixed precision if requested (disabled for now due to tensor dimension issues)
    if use_mixed_precision and False:  # Temporarily disable mixed precision
        policy = policy.half()
        print("✓ Using mixed precision (FP16)")
    else:
        print("✓ Using FP32 precision (mixed precision disabled)")

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

    out_path = os.path.join(output_dir, f'eval_log_{checkpoint.split("/")[-1]}.json')
    print("Saving log to %s" % out_path)
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
