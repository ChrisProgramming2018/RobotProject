import torch
import copy
from cnn_models import  CNNCritic
from tqc_models import Actor
import torch.nn as nn
import torch.nn.functional as F



# Building the whole Training Process into a class

class TQC(object):
    def __init__(self, state_dim, action_dim, actor_input_dim, args):
        input_dim = [args.history_length, args.size, args.size]
        
        self.actor = Actor(state_dim, action_dim, args).to(args.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), args.lr_actor)
        
        self.critic = CNNCritic(input_dim, state_dim, action_dim, args).to(args.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), args.lr_critic)
        self.target_critic = CNNCritic(input_dim, state_dim, action_dim, args).to(args.device)
        self.target_critic.load_state_dict(self.target_critic.state_dict())
        
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.tau = args.tau 
        self.device = args.device
        self.write_tensorboard = False
        self.top_quantiles_to_drop = args.top_quantiles_to_drop
	self.target_entropy = args.target_entropy
        self.quantiles_total = critic.n_quantiles * critic.n_nets
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=args.device)
        self.total_it = 0
        self.step = 0

    
    
    def train(self, replay_buffer, writer, iterations):
        self.step += 1
        if self.step % 1000 == 0:
            self.write_tensorboard = 1 - self.write_tensorboard
        for it in range(iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            
            obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(self.batch_size)
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
            alpha = torch.exp(self.log_alpha)
            with torch.no_grad(): 
                # Step 5: Get policy action
                new_next_action, next_log_pi = self.actor(next_state)

                # compute quantile at next state
                next_z = self.critic_target(next_state, new_next_action)
                sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
                sorted_z_part = sorted_z[:,:self.quantiles_total - self.top_quantiles_to_drop]


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

    def quantile_huber_loss_f(self, quantiles, samples):
        pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
        n_quantiles = quantiles.shape[2]
        tau = torch.arange(n_quantiles, device=self.device).float() / n_quantiles + 1 / 2 / n_quantiles
        loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
        return loss

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
