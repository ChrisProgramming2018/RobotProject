# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
 
import os
import sys 
import cv2
import time
import torch 
import random
import numpy as np
import robosuite as suite
from collections import deque
from datetime import datetime
from memory import ReplayBuffer
from agent import TD31v1
from torch.utils.tensorboard import SummaryWriter
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

def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname 
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def evaluate_policy(policy, writer, total_timesteps, args, env, episode = 10):
    """
    
    
    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """
    print("eval")
    size = 84
    use_gym = False
    avg_reward = 0.
    seeds = [x for x in range(episode)]
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        if use_gym:
            env.seed(s)
            obs = env.reset()
            done = False
        else:
            obs, done = env.reset(), False
            obs, state_buffer = stacked_frames(obs, size, args, policy)

        for step in range(args.max_episode_steps):
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            if not use_gym:
                obs, state_buffer = create_next_obs(obs, size, args, state_buffer, policy)
            avg_reward += reward * args.reward_scalling
    avg_reward /= len(seeds)
    writer.add_scalar('Evaluation reward', avg_reward, total_timesteps)
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print ("---------------------------------------")
    return avg_reward




def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')

def time_format(sec):
    """
    
    Args:
        param1():

    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)



def train(args, param):
    """

    Args:
    """
    
    # create CNN convert the [1,3,84,84] to [1, 200]
    torch.cuda.set_device(1)
    
    use_gym = False
    # in case seed experements
    args.seed = param
    now = datetime.now()    
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    #args.repeat_opt = repeat_opt
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pathname = str(args.env_name) + '-agent-' + str(args.policy)
    pathname += "_batch_size_" + str(args.batch_size) + "_lr_"+ str(args.lr)  
    pathname += '_update_freq: ' + str(args.target_update_freq) + "num_q_target_" + str(args.num_q_target) + "_seed_" + str(args.seed)
    pathname += "_actor_300_200"
    text = "Star_training target_update_freq: {}  num_q_target: {}  use device {} ".format(args.target_update_freq, args.num_q_target, args.device)
    print(pathname, text)
    write_into_file(pathname, text) 
    arg_text = str(args)
    write_into_file(pathname, arg_text) 
    tensorboard_name = 'runs/' + pathname 
    writer = SummaryWriter(tensorboard_name)
    
    if use_gym:
        env = gym.make(args.env_name)
        env.seed(args.seed)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        args.max_episode_steps = env._max_episode_steps
    else:
        size = 84
        env = suite.make(
                args.env_name,
                has_renderer=False,
                use_camera_obs=True,
                ignore_done=True,
                has_offscreen_renderer=True,
                camera_height=size,
                camera_width=size,
                render_collision_mesh=False,
                render_visual_mesh=True,
                camera_name='agentview',
                use_object_obs=False,
                camera_depth=True,
                reward_shaping=True,
                )
        
        

    state_dim = 200
    print("State dim, " , state_dim)
    action_dim = env.dof
    max_action = 1
    args.max_episode_steps = 200
    
    if args.policy == "TD3_ad":
        policy = TD31v1(state_dim, action_dim, max_action, args)
    elif args.policy == "DDPG":
        policy = DDPG(state_dim, action_dim, max_action, args)
        
    file_name = "./pytorch_models/{}".format(args.env_name)
    replay_buffer = ReplayBuffer()
    save_env_vid = False
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()
    scores_window = deque(maxlen=100) 
    episode_reward = 0
    evaluations = []
    tb_update_counter = 0
    while total_timesteps <  args.max_timesteps:
        tb_update_counter += 1
        # If the episode is done
        if done:
            episode_num += 1
            #env.seed(random.randint(0, 100))
            scores_window.append(episode_reward)
            average_mean = np.mean(scores_window)
            if tb_update_counter > args.tensorboard_freq:
                print("Write tensorboard")
                tb_update_counter = 0
                writer.add_scalar('Reward', episode_reward, total_timesteps)
                writer.add_scalar('Reward mean ', average_mean, total_timesteps)
                writer.flush()
            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                text = "Total Timesteps: {} Episode Num: {} ".format(total_timesteps, episode_num) 
                text += "Episode steps {} ".format(episode_timesteps)
                text += "Reward: {:.2f}  Average Re: {:.2f} Time: {}".format(episode_reward, np.mean(scores_window), time_format(time.time()-t0))
                
                print(text)
                write_into_file('search-' + pathname, text)
                #policy.train(replay_buffer, writer, episode_timesteps)
            # We evaluate the episode and we save the policy
            if total_timesteps > args.start_timesteps:
                policy.train(replay_buffer, writer, 200)
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq 
                evaluations.append(evaluate_policy(policy, writer, total_timesteps, args,  env))
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                save_model = file_name + '-{}reward_{:.2f}-agent{}'.format(episode_num, evaluations[-1], args.policy)
                policy.save(save_model)
            # When the training step is done, we reset the state of the environment
            if use_gym:
                obs = env.reset()
            else:
                state = env.reset()
                obs, state_buffer = stacked_frames(state, size, args, policy)

            # Set the Done to False
            done = False
            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
        # Before 10000 timesteps, we play random actions
        if total_timesteps < args.start_timesteps:
            if use_gym:
                action = env.action_space.sample()
            else:
                action = np.random.randn(env.dof)
        else: # After 10000 timesteps, we switch to the model
            if use_gym:
                action = policy.select_action(np.array(obs))
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if args.expl_noise != 0:
                    action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
            else:
                action = ( policy.select_action(np.array(obs)) + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

        
        if total_timesteps % args.target_update_freq == 0:
            if args.policy == "TD3_ad":
                policy.hardupdate()
        # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, done, _ = env.step(action)
        
        if not use_gym:
            new_obs, state_buffer = create_next_obs(new_obs, size, args, state_buffer, policy)
        # We check if the episode is done
        #done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        done_bool = 0 if episode_timesteps + 1 == args.max_episode_steps else float(done)
        if not use_gym:
            if episode_timesteps + 1 == args.max_episode_steps:
                done = True
        # We increase the total reward
        reward = reward * args.reward_scalling
        episode_reward += reward
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        if args.debug:
            print("add to buffer next_obs ", obs.shape)
            print("add to bufferobs ", new_obs.shape)
        replay_buffer.add((obs, new_obs, action, reward, done_bool))
        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        if total_timesteps > args.start_timesteps:
            policy.train(replay_buffer, writer, 0)
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


    # We add the last policy evaluation to our list of evaluations and we save our model
    evaluations.append(evaluate_policy(policy, writer, total_timesteps, args, episode_num))
