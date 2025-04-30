import gymnasium as gym
import torch
import mani_skill.envs
import numpy as np


env = gym.make("PushT-v1", obs_mode="state", control_mode="pd_ee_delta_pose") # Match your settings
obs, info = env.reset(seed=512)
# agent_qpos = env.unwrapped.agent.robot.get_qpose() 
# tee_pose = env.unwrapped.tee.pose
state = env.unwrapped.get_state_dict()
torch.save(state, "pusht_initial_state512.pth")
env.close()