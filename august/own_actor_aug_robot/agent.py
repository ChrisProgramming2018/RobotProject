import torch
import copy
from cnn_models import Actor, CNNCritic

import torch.nn as nn
import torch.nn.functional as F




# Building the whole Training Process into a class

class TD31v1(object):
    def __init__(self, state_dim, action_dim, actor_input_dim, args):
        input_dim = [args.history_length, args.size, args.size]
        self.actor = Actor(state_dim, action_dim).to(args.device)
        self.actor_target = Actor(state_dim, action_dim).to(args.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), args.lr_actor)
        self.critic = CNNCritic(input_dim, state_dim, action_dim, args).to(args.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), args.lr_critic)
        self.target_critic = CNNCritic(input_dim, state_dim, action_dim, args).to(args.device)
        self.target_critic.load_state_dict(self.target_critic.state_dict())
        self.list_target_critic = []
        for c in range(args.num_q_target):
            critic_target = CNNCritic(input_dim, state_dim, action_dim, args).to(args.device)
            critic_target.load_state_dict(critic_target.state_dict())
            self.list_target_critic.append(critic_target)
        self.num_q_target = args.num_q_target
        self.max_action = 1
        self.update_counter = 0
        self.currentQNet = 0
        self.step = 0 
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.tau = args.tau 
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.device = args.device
        self.actor_clip_gradient = args.actor_clip_gradient
        self.write_tensorboard = False
    def select_action(self, state):
        state = torch.Tensor(state).to(self.device).div_(255)
        state = state.unsqueeze(0)
        state = self.critic.create_vector(state)
        return self.actor(state).cpu().data.numpy().flatten()
    
    
    def train(self, replay_buffer, writer, iterations):
        self.step += 1
        if self.step % 1000 == 0:
            self.write_tensorboard = 1 - self.write_tensorboard
        for it in range(iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(self.batch_size)
            #batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(self.batch_size)
            #state_image = torch.Tensor(batch_states).to(self.device).div_(255)
            #next_state = torch.Tensor(batch_next_states).to(self.device).div_(255)
            # create vector 
            #reward = torch.Tensor(batch_rewards).to(self.device)
            #done = torch.Tensor(batch_dones).to(self.device)
            obs = obs.div_(255)
            next_obs = next_obs.div_(255)
            obs_aug = obs_aug.div_(255)
            next_obs_aug = next_obs_aug.div_(255)

            state = self.critic.create_vector(obs)
            detach_state = state.detach()
            state_aug = self.critic.create_vector(obs_aug)
            next_state = self.target_critic.create_vector(next_obs)
            detach_state_aug = state_aug.detach()
            next_state_aug = self.target_critic.create_vector(next_obs_aug)
            with torch.no_grad(): 
                # Step 5: From the next state s’, the Actor target plays the next action a’
                next_action = self.actor_target(next_state)
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                
                # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
                target_Q = 0
                for critic in self.list_target_critic:
                    
                    target_Q1, target_Q2 = critic(critic.create_vector(next_obs), next_action) 
                    target_Q += torch.min(target_Q1, target_Q2)
                target_Q *= 1./ self.num_q_target  
                # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
                target_Q = reward + (not_done * self.discount * target_Q).detach()
                # again with augmented data
                next_action_aug = self.actor_target(next_state_aug)    
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action_aug = (next_action_aug + noise).clamp(-self.max_action, self.max_action)
                
                target_aug_Q = 0
                for idx, critic in enumerate(self.list_target_critic):
                    target_Q1, target_Q2 = critic(critic.create_vector(next_obs_aug), next_action_aug) 
                    target_aug_Q_min = torch.min(target_Q1, target_Q2)
                    if self.write_tensorboard:
                        writer.add_scalar('Critic-{} q'.format(idx), target_aug_Q_min.mean(), self.step)

                    target_aug_Q += target_aug_Q_min

                target_aug_Q *= 1./ self.num_q_target  
               
                target_aug_Q = reward + (not_done * self.discount * target_aug_Q).detach()

                target_Q = (target_Q + target_aug_Q) / 2.



            current_Q1, current_Q2 = self.critic(state, action) 

            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            if self.write_tensorboard:
                writer.add_scalar('Critic loss', critic_loss, self.step)
            
            # again for augment
            Q1_aug, Q2_aug = self.critic(state_aug, action) 
            critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(Q2_aug, target_Q)
            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % self.policy_freq == 0:
                # print("cuurent", self.currentQNet)
                obs = replay_buffer.sample_actor(self.batch_size)
                obs = obs.div_(255)
                state = self.critic.create_vector(obs)
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                if self.write_tensorboard:
                    writer.add_scalar('Actor loss', actor_loss, self.step)
                self.actor_optimizer.zero_grad()
                #actor_loss.backward(retain_graph=True)
                actor_loss.backward()
                # clip gradient
                # self.actor.clip_grad_value(self.actor_clip_gradient)
                torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.actor_clip_gradient)
                self.actor_optimizer.step()
                
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    
                
    def hardupdate(self):
        self.update_counter += 1
        self.currentQNet = self.update_counter % self.num_q_target
        for param, target_param in zip(self.target_critic.parameters(), self.list_target_critic[self.currentQNet].parameters()):
            target_param.data.copy_(param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
                
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor) 
