import os
import sys
import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import robosuite as suite
from agent import TD31v1
import cv2
from PIL import Image


def create_next_obs(next_obs, size, args, state_buffer, policy, debug=False):
    state = next_obs["image"]
    state = state[:,:,0]
    state = torch.tensor(state, dtype=torch.int8, device=args.device)
    state_buffer.append(state)
    state = torch.stack(list(state_buffer), 0)
    state = state.cpu()
    obs = np.array(state)
    return obs, state_buffer 


def stacked_frames(state, size, args, policy, debug=False):
    state = state["image"]
    if debug:
        img = Image.fromarray(state, 'RGB')
        img.save('my.png')
        img.show()
        img = Image.fromarray(lum_img, 'L')
        img.save('my_gray.png')
    state = state[:,:,0]
    state = torch.tensor(state, dtype=torch.int8, device=args.device)
    zeros = torch.zeros_like(state)
    state_buffer = deque([], maxlen=args.history_length)
    state_buffer.append(zeros)
    state_buffer.append(zeros)
    state_buffer.append(zeros)
    state_buffer.append(state)
    state = torch.stack(list(state_buffer), 0)
    state = state.cpu()
    obs = np.array(state)
    return obs, state_buffer

def get_state(state, size, args, perception, debug=False):
    state = state["image"]
    if debug:
        img = Image.fromarray(state, 'RGB')
        img.save('my.png')
        img.show()
        img = Image.fromarray(lum_img, 'L')
        img.save('my_gray.png')
    state = state[:,:,0]
    state = torch.tensor(state, dtype=torch.int8, device=args.device)
    zeros = torch.zeros_like(state)
    state_buffer = deque([], maxlen=args.history_length)
    state_buffer.append(zeros)
    state_buffer.append(zeros)
    state_buffer.append(zeros)
    state_buffer.append(state)
    state = torch.stack(list(state_buffer), 0)
    obs = np.array(state)
    return obs, state_buffer



def main(args):
    """ Starts different tests

    Args:
        param1(args): args

    """
    size = args.size
    random_agent = True
    random_agent = False
    use_render = False
    # use_render = True
    print("use render {} ".format(use_render))
    env = suite.make(
            args.env_name,
            has_renderer=use_render,
            use_camera_obs=True,
            ignore_done=True,
            has_offscreen_renderer=True,
            camera_height=size,
            camera_width=size,
            render_collision_mesh=use_render,
            render_visual_mesh=True,
            camera_name='agentview',
            use_object_obs=False,
            camera_depth=False,
            reward_shaping=True,
            )
    state = env.reset()
    state_dim = 200
    action_dim = env.dof
    max_action = float(1)
    min_action = float(-1)
    width = size 
    height = size
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 30
    video_filename = 'output.avi'
    video= cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    policy = TD31v1(state_dim, action_dim, max_action, args) 
    directory = "pytorch_models/"
    filename ="SawyerLift-701reward_75.21-agentTD3_ad"
    filename ="SawyerLift-5401reward_3531.73-agentTD3_ad"
    filename ="SawyerLift-7201reward_3187.32-agentTD3_ad"
    filename = directory + filename
    print("Load " , filename)
    if not random_agent:
        policy.load(filename)
    avg_reward = 0.
    seeds = [x for x in range(args.repeat)]
    episode = 1
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        print("iteration ", s)
        obs = env.reset()
        obs, state_buffer = stacked_frames(obs, size, args, policy)
        done = False
        for x in range(args.timesteps):
            action = np.random.randn(env.dof)
            if not random_agent:
                action = policy.select_action(np.array(obs))
            obs , reward, done, _ = env.step(action)
            img = obs["image"]
            gray = cv2.normalize(img, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            name = "images/state-{}.jpg".format(x+1000)
            im = Image.fromarray(img)
            im.save(name)
            frame = cv2.merge([gray, gray, gray])
            video.write(frame)
            obs, state_buffer = create_next_obs(obs, size, args, state_buffer, policy)
            avg_reward += reward * 10
            if use_render:
                time.sleep(0.02)
                env.render()
        print("episode reward {}".format(avg_reward/episode))
        episode += 1
    avg_reward /= len(seeds)
    cv2.destroyAllWindows()
    video.release()
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print ("---------------------------------------")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="SawyerLift", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--timesteps', default=200, type=int)
    parser.add_argument('--eval_freq', default=10000, type=int)  # How often the evaluation step is performed (after how many timesteps)
    parser.add_argument('--repeat', default=1, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--max_timesteps', default=2e6, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr-critic', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr-actor', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--save_model', default=True, type=bool)     # Boolean checker whether or not to save the pre-trained model
    parser.add_argument('--expl_noise', default=0.1, type=float)      # Exploration noise - STD value of exploration Gaussian noise
    parser.add_argument('--batch_size', default= 256, type=int)      # Size of the batch
    parser.add_argument('--discount', default=0.99, type=float)      # Discount factor gamma, used in the calculation of the total discounted reward
    parser.add_argument('--tau', default=0.005, type= float)        # Target network update rate
    parser.add_argument('--policy_noise', default=0.2, type=float)   # STD of Gaussian noise added to the actions for the exploration purposes
    parser.add_argument('--noise_clip', default=0.5, type=float)     # Maximum value of the Gaussian noise added to the actions (policy)
    parser.add_argument('--policy_freq', default=2, type=int)         # Number of iterations to wait before the policy network (Actor model) is updated
    parser.add_argument('--target_update_freq', default=50, type=int)
    parser.add_argument('--num_q_target', default=4, type=int)    # amount of qtarget nets
    parser.add_argument('--train_every_step', default=True, type=bool)    # amount of qtarget nets
    parser.add_argument('--tensorboard_freq', default=5000, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--device', default='cuda', type=str)    # amount of qtarget nets
    parser.add_argument('--run', default=1, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--agent', default="301", type=str)    # load the weights saved after the given number 
    parser.add_argument('--reward_scalling', default=10, type=int)    # amount
    parser.add_argument('--policy', default="TD3_ad", type=str)     # Maximum value of the Gaussian noise added to the actions (policy)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--image_pad', default=4, type=int)     #
    parser.add_argument('--actor_clip_gradient', default=1., type=float)     # Maximum value of the Gaussian noise added to
    parser.add_argument('--locexp', type=str)     # Maximum value
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--no_render', type=bool)
    parser.add_argument('--size', default=84, type=int)
    arg = parser.parse_args()
    main(arg)
