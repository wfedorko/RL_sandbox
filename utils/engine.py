from models.models import SimpleCNNDuelingAllMean

import gym
import math
import time
import torch
import numpy as np

import os

from agents.QNet_Agent import QNet_Agent
from utils.ExperienceReplay import ExperienceReplay
from utils.training_helpers import calculate_epsilon
from utils.plotting_utils import plot_results

import random


from baselines.common.atari_wrappers import make_atari, wrap_deepmind

class Engine:
    """The training engine 
    
    Performs training and evaluation
    """

    def __init__(self, config):
        
        self.config=config
        
        self.env=make_atari(config.env_id)
        self.env=wrap_deepmind(self.env)
        config.number_of_outputs=self.env.action_space.n
        
        self.model = SimpleCNNDuelingAllMean(config)
        
        
        self.use_cuda = torch.cuda.is_available() and config.use_cuda
        self.device=torch.device(config.device if config.use_cuda else "cpu")
        
        dirName='{}/{}'.format(config.result_path,config.exp_name)
        
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists")   
        
        if config.save_movies:
            print('will save movies')
            self.env = gym.wrappers.Monitor(self.env, '{}/{}/mp4'.format(config.result_path,config.exp_name),
                                            video_callable=lambda episode_id: episode_id%20==0,
                                            force=True)
            
        if hasattr(config,'rand_seed'):
            self.env.seed(config.rand_seed)
            torch.manual_seed(config.rand_seed)
            random.seed(config.rand_seed)
        
        self.memory=ExperienceReplay(config.replay_memory_size)
        
        self.qnet_agent=QNet_Agent(config, self.model, self.memory, self.device)
        
    #def calculate_epsilon(self):
    #    self.epsilon = egreedy_final + (self.config.egreedy - self.config.egreedy_final) * \
    #              math.exp(-1. * self.steps_done / self.config.egreedy_decay )
        
        
    def train_agent(self):
        
        steps_total=np.full([self.config.num_episodes],-999,dtype=np.int32)
        reward_total=np.full([self.config.num_episodes],-999,dtype=np.int32)

        frames_total=0

        solved_after = 0
        solved = False

        start_time = time.time()

        for i_episode in range(self.config.num_episodes):

            state = self.env.reset()
            #for step in range(100):
            step=0
            reward_total[i_episode]=0

            while True:

                step+=1
                frames_total += 1

                epsilon=calculate_epsilon(frames_total,self.config)

                #action=env.action_space.sample()
                action=self.qnet_agent.select_action(state,epsilon)

                new_state, reward, done, info = self.env.step(action)
                self.memory.push(state, action, new_state,
                                 reward, done)

                reward_total[i_episode]+=reward

                self.qnet_agent.optimize()

                state=new_state


                if done:
                    steps_total[i_episode]=step

                    if i_episode>100:
                        mean_reward_100 = np.sum(reward_total[i_episode-100:i_episode])/100


                        if (mean_reward_100 > self.config.score_to_solve and solved == False):
                            print("SOLVED! After %i episodes " % i_episode)
                            solved_after = i_episode
                            solved = True

                    if (i_episode % self.config.report_interval == 0 and i_episode>1):

                        plot_results(reward_total, i_episode,self.config)

                        print("**** Episode  {} **** ".format(i_episode))
                        recent_avg_reward=np.average(reward_total[i_episode-self.config.report_interval:i_episode])
                        print("Recent average reward: {}".format(recent_avg_reward))
                        if i_episode>100:
                            print("Reward over last 100: {}".format(mean_reward_100))
                        full_avg_so_far=np.average(reward_total[:i_episode])
                        print("Average over all episodes so far: {}".format(full_avg_so_far))
                        print("epsilon: {}".format(epsilon))
                        elapsed_time = time.time() - start_time
                        print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

                        #print("Episode {} finished after: {}".format(i_episode,step))
                    break

        if solved:
            print("Solved after %i episodes" % solved_after)