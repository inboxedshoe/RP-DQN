import math
import numpy
import matplotlib.pyplot as plt
import torch
import collections

class Scheduler():
    def __init__(self, start, end, total_episodes, decay_type, warmup_episodes = 0, linear_percentage = 0.5):
        assert decay_type == 'exp' or decay_type == 'lin' or decay_type == 'const'
        self.decay_type = decay_type
        self.warmup_episodes = warmup_episodes
        self.start = start
        self.end = end
        
        if decay_type == 'exp':
            self.decay = ((total_episodes - warmup_episodes)*0.9)/(4.5-(start - end))
        elif decay_type == 'lin':
            self.lin_episode_end = warmup_episodes + linear_percentage * (total_episodes - warmup_episodes)
            self.decay = (end-start)/(linear_percentage * (total_episodes - warmup_episodes))
            
            
    def get_exploration_rate(self, episode):
        if episode > self.warmup_episodes:
            if self.decay_type == 'exp':
                return self.end + (self.start - self.end) * math.exp(-1. * (episode- self.warmup_episodes) / self.decay)
            elif self.decay_type == 'lin':
                if self.lin_episode_end > episode:
                    return self.decay*(episode- self.warmup_episodes) + self.start
                else:
                    return self.end
            elif self.decay_type == 'const':
                return self.end
        else:
            return self.start

class Logging():
    
    def __init__(self, writer, log_opts, model):
        self.writer = writer
        self.log_opts = log_opts
        self.model = model
        self.best_val_loss = math.inf
        
    def log_learn(self, loss, y, state_Qs, steps_done):
        """logs q values, targets and the q loss in the learn method every n steps done"""
        #tensorboard
        if (steps_done%self.log_opts['log_step_frequency'] == 0 and self.writer is not None):
            if self.log_opts['log_q_loss']:
                self.writer.add_scalar('Q-learning Loss', loss, steps_done)
            if self.log_opts['log_target']:
                self.writer.add_scalar('Target Mean', y.mean(), steps_done)
            if self.log_opts['log_qs']:
                self.writer.add_scalar('Q-Values Mean', state_Qs.mean(), steps_done)
                self.writer.add_scalar('Q-Values std', state_Qs.std(), steps_done)
                
        #verbose
        if steps_done%self.log_opts['log_step_frequency'] == 0 and self.log_opts['verbose']:
            if self.log_opts['log_q_loss']:
                print('Q-learning Loss: ', loss)
            if self.log_opts['log_target']:
                print('Target Mean', y.mean())
            if self.log_opts['log_qs']:
                print('Q-Values Mean', state_Qs.mean())
                print('Q-Values std', state_Qs.std())
                
    def log_episode(self, cost, cap, episodes_done, exploration_rate):
        #logging to tensorboard
        if self.writer is not None:
            #log buffer cap
            if (episodes_done%self.log_opts['log_episode_frequency']==0 and self.log_opts['log_buffer_cap']):
                self.writer.add_scalar("Buffer Capacity filled in %", cap*100 , episodes_done)
                
            # Log epsilon value
            if (episodes_done%self.log_opts['log_episode_frequency']==0 and self.log_opts['log_epsilon']):
                self.writer.add_scalar("Epsilon", exploration_rate, episodes_done)
                    
            # logging Train cost to tensorboard
            if (episodes_done%self.log_opts['log_episode_frequency'] == 0 and self.log_opts['log_mean_train'] and cost is not None):
                self.writer.add_scalar("Mean Cost Train",torch.mean(cost),episodes_done)
            
            #log images
            if (episodes_done%self.log_opts['log_val_frequency'] == 0 and self.log_opts['log_val_plots']):
                self.model.write_n_plots(10,tboard=False,file=True)
            
            #log val cost
            if (episodes_done%self.log_opts['log_val_frequency'] == 0 and self.log_opts['log_mean_val']):
                tours, val_cost = self.model.validate()
                mean_cost = torch.mean(val_cost)
                if mean_cost < self.best_val_loss:
                     self.best_val_loss = mean_cost
                    
                self.writer.add_scalar("Mean Cost Validation", mean_cost, episodes_done)
                
        # verbose
        if self.log_opts['verbose']:
        #log buffer cap
            if (episodes_done%self.log_opts['log_episode_frequency']==0 and self.log_opts['log_buffer_cap']):
                print("Buffer Capacity filled in %", cap*100)
                
            # Log epsilon value
            if (episodes_done%self.log_opts['log_episode_frequency']==0 and self.log_opts['log_epsilon']):
                print("Epsilon", exploration_rate)
                    
            # logging Train cost to tensorboard
            if (episodes_done%self.log_opts['log_episode_frequency'] == 0 and self.log_opts['log_mean_train'] and cost is not None):
                print("Mean Cost Train",torch.mean(cost))
            
            #log val cost
            if (episodes_done%self.log_opts['log_val_frequency'] == 0 and self.log_opts['log_mean_val']):
                tours, val_cost = self.model.validate()
                print("Mean Cost Validation", torch.mean(val_cost))
                
    def log_target_and_reward_distributions(self, steps_done, states, rewards, y):
        if (steps_done%self.log_opts['log_step_debug_frequency'] == 0 and self.writer is not None and self.log_opts['debug']):
        
            newi = states.i.squeeze(1).detach().cpu().numpy()
            newy = y.detach().cpu().numpy()
            fig = plt.figure()
            plt.hist2d(newi,newy)
            self.writer.add_figure('Target Distribution', fig, steps_done)
            
            i_s = []
            vals = []
            vals2 = []
            for step in states.i.unique():
                i_s.append(step.item())
                vals.append((y.unsqueeze(1)[states.i == step]).mean().item())
                vals2.append((rewards.unsqueeze(1)[states.i == step]).mean().item())
                
                # self.writer.add_histogram("Target at step {}".format(int(step.item())), y.unsqueeze(1)[states.i == step] ,steps_done)
                # self.writer.add_histogram("rewards at step {}".format(int(step.item())), rewards.unsqueeze(1)[states.i == step] ,steps_done)
            self.writer.add_histogram("step index", states.i ,steps_done)
            
            fig = plt.figure()
            plt.plot(i_s,vals)
            self.writer.add_figure('Target Means per step', fig, steps_done)
            
            fig = plt.figure()
            plt.plot(i_s,vals2)

            self.writer.add_figure('Reward Means per step', fig, steps_done)

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# episodes = 100
# eps = Epsilon(0.9, 0.1, episodes, 'const', warmup_episodes=10)
# exp_rate = []
# for i in range(episodes):
#     exp_rate.append(eps.get_exploration_rate(i))
    
# plt.plot(exp_rate)
# plt.show()