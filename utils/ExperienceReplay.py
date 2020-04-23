import numpy as np
import math
import random

from utils.processing_utils import preprocess_frame
from utils.processing_utils import transpose_tuple

class ExperienceReplay():
    def __init__(self, memory_device, state_shape, state_dtype, action_dtype):
        self.capacity=config.replay_memory_size
        self.memory_state_shape=[self.capacity]
        for i in transpose_tuple:
            self.memory_state_shape+=state_shape[i]
        print('creating state memory with shape: {}'.format(self.memory_state_shape))
        
        self.memory_state=torch.zeros(self.memory_state_shape,
                                      dtype=state_dtype).to(memory_device)
        self.memory_new_state=torch.zeros(self.memory_state_shape,
                                          dtype=state_dtype).to(memory_device)
        self.memory_action=torch.zeros([self.capacity, 1],
                                       dtype=action_dtype).to(memory_device)
        self.memory_reward=torch.zeros([self.capacity],
                                       dtype=torch.float32).to(memory_device)
        self.memory_done=torch.zeros([self.capacity],
                                     dtype=torch.long).to(memory_device)
        
        self.filled_to=0
        self.position=0
        
        self.state_dtype=state_dtype
        self.action_dtype=action_dtype
        
        #this is a dummy equal weights vector to do random samples
        self.wgt=torch.ones(self.capacity,dtype=torch.float32).to(memory_device)
        
        
    
    def push(self, state,
             action, new_state,
             reward, done):
        
            self.memory_state[self.position,:]=state
            self.memory_new_state[self.position,:]=new_state
            self.memory_action[self.position,0]=action
            self.memory_reward[self.position]=reward
            self.memory_done[self.position]=done
            
              
            self.position=(self.position+1)%self.capacity
            self.filled_to=min(self.capacity,self.filled_to+1)
        
    
    def sample(self,batch_size):
        
        #idx=torch.multinomial(self.wgt,batch_size)
        idx=torch.randint(0,self.filled_to,(batch_size,),dtype=torch.long,device=device)
        return (self.memory_state[idx],
                self.memory_action[idx],
                self.memory_new_state[idx],
                self.memory_reward[idx],
                self.memory_done[idx])
        
    def __len__(self):
        return self.filled_to


