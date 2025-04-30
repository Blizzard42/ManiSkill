import os
import time
from typing import Optional
import numpy as np
import torch
import tyro
from tqdm import tqdm
from collections import defaultdict
import random
from dataclasses import dataclass

from train import Agent, Args as TrainArgs
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils import common

@dataclass
class VisualizeArgs():
    checkpoint_path: str
    """Where the model is saved"""
    output_dir: str = "runs/visualizations"
    """Where to save the video to"""
    use_fixed_start_state: bool = False
    """If toggled, try to load and use the initial_qpos specified by --initial_qpos_path."""
    initial_state_path: Optional[str] = None
    """Path to a .npy file containing the desired initial qpos array for the environment.
       If set along with --use-fixed-start-pose, this qpos will be used for reset."""
    no_video: bool = False
    """Controls whether to record videos"""
    seed: int = 42  # overrides TrainArgs.seed for eval
    """Seed to use"""
    eval_seed: Optional[int] = None
    """Seed to use for resetting evaluation environments *specifically for each episode*.
       If set, this seed (potentially offset per parallel env/episode) will be passed to env.reset().
       Useful for testing determinism or slight variations from the same start state."""
    num_episodes: int = 10
    """The number of episodes to execute"""
    max_episode_steps: Optional[int] = 100
    """Maximum number of steps per episode, should be 100 based on push T environment"""
    num_eval_envs: int = 10
    """Number of simultaneous envs to run"""
    act_horizon: int = 8
    """Should match act_horizon the model was trained on"""

def visualize(args: VisualizeArgs, checkpoint_path: str, num_episodes: int, output_dir: str):
    """
    Loads a trained diffusion policy model and runs evaluations, saving videos.

    Args:
        args: Visualization Arguments (loaded or defined). Must match the training run.
        checkpoint_path: Path to the saved model checkpoint (.pt file).
        num_episodes: Number of episodes to visualize.
        output_dir: Directory to save the evaluation videos.
    """

    # --- Setup ---
    args.num_eval_envs = min(num_episodes, 10, args.num_eval_envs) # Limit parallel envs for smoother video recording if needed

    print("Using arguments:")
    print(f"  checkpoint_path: {checkpoint_path}")
    print(f"  output_dir: {output_dir}")

    # SEEDING
    random.seed(args.seed) # Use a different seed for evaluation
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # LOAD AND SET INITIAL STATE
    loaded_state_dict = None
    replicated_state_dict = None # State dictionary replicated for all parallel envs
    if args.use_fixed_start_state: # Check the correct flag name
        if args.initial_state_path is None:
            print("\nWarning: --use-fixed-start-state is True, but --initial_state_path is not set. Will use default environment reset randomization.\n")
        else:
            try:
                print(f"\nLoading initial state dictionary from: {args.initial_state_path}")
                # --- Load the dictionary using torch.load ---
                loaded_state_dict = torch.load(args.initial_state_path, map_location=device)
                # ---

                print(f"Successfully loaded initial state dict with keys: {list(loaded_state_dict.keys())}")

                # --- Replicate dictionary values ---
                replicated_state_dict = dict()
                for key, value in loaded_state_dict.items():
                    if isinstance(value, torch.Tensor):
                        current_val = value
                        if value.shape and value.shape[0] == 1: # Check shape exists and batch dim is 1
                           current_val = value.squeeze(0)
                        # Add new batch dim and repeat
                        replicated_state_dict[key] = current_val.unsqueeze(0).repeat_interleave(args.num_eval_envs, dim=0)
                    elif isinstance(value, np.ndarray):
                         current_val = value
                         if value.shape and value.shape[0] == 1:
                            current_val = value.squeeze(0)
                         current_tensor = torch.from_numpy(current_val).to(device)
                         replicated_state_dict[key] = current_tensor.unsqueeze(0).repeat_interleave(args.num_eval_envs, dim=0)
                    else:
                        # Assume non-tensor/array data can be copied directly
                        replicated_state_dict[key] = value
                # ---

                print(f"Replicated loaded state dictionary for {args.num_eval_envs} environments.")

            except FileNotFoundError:
                print(f"\nError: Initial state file not found at {args.initial_state_path}. Will use default environment reset randomization.\n")
                replicated_state_dict = None
            except Exception as e:
                print(f"\nError loading or replicating initial state from {args.initial_state_path}: {e}. Will use default environment reset randomization.\n")
                import traceback
                traceback.print_exc()
                replicated_state_dict = None
    # --- End Load/Set Initial State ---

    # --- Environment Setup ---
    video_dir = os.path.abspath(output_dir)
    os.makedirs(video_dir, exist_ok=True)
    print(f"Saving videos to: {video_dir}")

    env_kwargs = dict(
        control_mode="pd_ee_delta_pose",
        reward_mode="sparse", # Reward mode doesn't matter much for visualization
        obs_mode="state",
        render_mode="rgb_array" if not args.no_video else "none", # Crucial for video recording
        human_render_camera_configs=dict(shader_pack="default") # Optional: nice rendering
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    else:
        # Try to infer from env_id if not provided, though it should be in args
        # from mani_skill.envs.registration import REGISTERED_ENVS
        # if args.env_id in REGISTERED_ENVS:
        #     env_kwargs["max_episode_steps"] = REGISTERED_ENVS[args.env_id].cls.max_episode_steps
        print("Warning: max_episode_steps not explicitly set in args, using environment default.")


    other_kwargs = dict(obs_horizon=2)
    eval_envs = make_eval_envs(
        "PushT-v1",
        args.num_eval_envs,
        "physx_cuda",
        env_kwargs,
        other_kwargs,
        video_dir=video_dir if not args.no_video else None
    )
    print(f"Created {args.num_eval_envs} evaluation environments.")

    # --- Agent Setup ---
    agent = Agent(eval_envs, TrainArgs(obs_horizon=2, act_horizon=4, pred_horizon=8)).to(device) # Use eval_envs to get space info, args for model structure

    # --- Load Checkpoint ---
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # It's generally recommended to evaluate using the EMA weights
    if 'ema_agent' in checkpoint:
        print("Loading EMA agent state dict.")
        agent.load_state_dict(checkpoint['ema_agent'])
    elif 'agent' in checkpoint:
        print("Warning: EMA agent not found in checkpoint, loading base agent state dict.")
        agent.load_state_dict(checkpoint['agent'])
    else:
        raise ValueError("Checkpoint does not contain 'ema_agent' or 'agent' state dict.")

    agent.eval() # Set agent to evaluation mode

    # --- Evaluation Loop (adapted from diffusion_policy.evaluate.evaluate) ---
    pbar = tqdm(total=num_episodes)
    episodes_run = 0
    eval_metrics = defaultdict(list) # Store metrics if needed

    while episodes_run < num_episodes:
        try:
            # RESETTING ENVIRONMENT
            reset_seeds = None
            if args.eval_seed is not None:
                # Generate slightly different seeds for each parallel env within this reset cycle,
                # based on the main eval_seed and how many episodes have already run.
                # This ensures different seeds if num_episodes > num_eval_envs.
                reset_seeds = [args.eval_seed + episodes_run + i for i in range(args.num_eval_envs)]

            print(f"Resetting envs for episode {episodes_run+1} onwards. Using seed={reset_seeds}")
            obs, info = eval_envs.reset(seed=reset_seeds)

            if replicated_state_dict is not None:
                print("Setting fixed initial state dictionary...")
                try:
                    # --- Explicitly call set_state_dict ---
                    eval_envs.call("set_state_dict", replicated_state_dict)
                    # ---
                    print("Successfully set fixed initial state.")
                    single_obs = eval_envs.call("get_obs")
                    obs = single_obs.unsqueeze(1).repeat(1, 2, 1)
                except Exception as e:
                     print(f"\nError calling set_state_dict: {e}. Continuing with state after reset.\n")
                     import traceback
                     traceback.print_exc()

            # MAIN TRAINING LOOP
            terminated = truncated = np.zeros(args.num_eval_envs, dtype=bool)
            while not (terminated.any() or truncated.any()):
                with torch.no_grad():
                    # Prepare observation tensor
                    obs_tensor = common.to_tensor(obs, device)
                    # Get action sequence from policy
                    action_seq = agent.get_action(obs_tensor) # (B, act_horizon, act_dim)

                # Execute action sequence
                # The original evaluate function executes act_horizon steps per policy call
                for i in range(args.act_horizon):
                    # Ensure action is on the correct device/format for the environment
                    action_step = action_seq[:, i]
                    action_np = action_step # Keep as tensor if env handles it

                    # Step environment
                    obs, rew, terminated, truncated, info = eval_envs.step(action_np)

                    # If any environment finishes, break the inner loop to reset them
                    if terminated.any() or truncated.any():
                        break

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                episodes_run += eval_envs.num_envs
                pbar.update(eval_envs.num_envs)

        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            import traceback
            traceback.print_exc()
            break # Stop evaluation if something goes wrong

    pbar.close()
    eval_envs.close()
    print("Evaluation finished.")

    # --- Print Metrics (Optional) ---
    if eval_metrics:
        print("\n--- Evaluation Metrics ---")
        for k in eval_metrics.keys():
            metric_array = np.concatenate(eval_metrics[k]) # Flatten list of arrays/scalars
            print(f"{k}: Mean={np.mean(metric_array):.4f}, Std={np.std(metric_array):.4f} (over {len(metric_array)} episodes)")

    print(f"\nVideos saved in: {video_dir}")

def main():
    # --- Argument Parsing ---
    args = tyro.cli(VisualizeArgs)

    # --- Run Visualization ---
    visualize(args, args.checkpoint_path, args.num_episodes, args.output_dir)

if __name__ == "__main__":
    main()