import math
import gymnasium as gym
import torch
import mani_skill.envs
import numpy as np
import matplotlib.pyplot as plt

def adjust_quaternion_by_degrees(quaternion, degrees):
    wo, xo, yo, zo = quaternion
    # compute delta
    theta = degrees * math.pi/180
    qd = [math.cos(theta/2), 0., 0., math.sin(theta/2)]
    # multiply (delta ⊗ old)
    wn = qd[0]*wo - qd[1]*xo - qd[2]*yo - qd[3]*zo
    xn = qd[0]*xo + qd[1]*wo + qd[2]*zo - qd[3]*yo
    yn = qd[0]*yo - qd[1]*zo + qd[2]*wo + qd[3]*xo
    zn = qd[0]*zo + qd[1]*yo - qd[2]*xo + qd[3]*wo
    # normalize
    norm = math.sqrt(wn*wn + xn*xn + yn*yn + zn*zn)
    return torch.tensor([wn, xn, yn, zn]) / norm

env = gym.make("PushT-v1", obs_mode="rgb", control_mode="pd_ee_delta_pose", )
obs, info = env.reset(seed=333)
img_np = obs['sensor_data']['base_camera']['rgb'].squeeze(0).cpu().numpy()

# SHOW INITIAL IMAGE
plt.imshow(img_np)
plt.axis('off')
plt.show()

state = env.unwrapped.get_state_dict()
print(state['actors']['Tee'])
# Adjust T rotation:
# breakpoint()
state['actors']['Tee'][0,3:7] = adjust_quaternion_by_degrees(state['actors']['Tee'][0,3:7], -45)
# Adjust T position
state['actors']['Tee'][0][0] += 0.2
state['actors']['Tee'][0][1] += 0.1
env.unwrapped.set_state_dict(state)
new_obs = env.unwrapped.get_obs()

img_np = new_obs['sensor_data']['base_camera']['rgb'].squeeze(0).cpu().numpy()
plt.imshow(img_np)
plt.axis('off')
plt.show()

torch.save(state, "pusht_initial_state_controlled2.pth")
env.close()
