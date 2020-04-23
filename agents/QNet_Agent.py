import torch
import copy
from utils.ExperienceReplay import ExperienceReplay
from utils.processing_utils import preprocess_frame
import os

import random

class QNet_Agent():
    def __init__(self, config, model, memory, device):
        
        self.config=config
        
        self.device=device
        
        self.nn = model.to(device) 
        self.target_nn = copy.deepcopy(self.nn) 
        
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.nn.parameters(),
                                          lr=config.learning_rate)
        
        self.memory=memory
        
        self.number_of_frames = 0

        self.file2save='{}/{}/model.pth'.format(self.config.result_path,
                                                self.config.exp_name)

        
        if config.resume_previous_training:
            self.load_model()
        
    def select_action(self,state,epsilon):
        
        random_for_egreedy=torch.rand(1).item()
        
        if random_for_egreedy>epsilon:
            self.nn.eval()
            with torch.no_grad():
                state=state.to(self.device)
                predicted_value_from_nn=self.nn(state).squeeze()
                #print('predicted value from nn:')
                #print(predicted_value_from_nn)
                action=torch.argmax(predicted_value_from_nn).item()
                #print('action: {}'.format(action))
        else:
            # -1 is deliberate - randint gives numbers within bounds INCLUSIVE
            action=random.randint(0,self.config.number_of_outputs-1)
                
                
        return action
    
    def load_model(self):
        
        if os.path.exists(self.file2save):
        
            print('loading previous model')
            self.nn.load_state_dict(torch.load(self.file2save))

    def save_model(self):
        torch.save(self.nn.state_dict(), self.file2save)
    
    def optimize(self):
        
        if len(self.memory)<self.config.batch_size:
            return
        
        state, action, new_state, reward, done = self.memory.sample(self.config.batch_size)
        
        state=state.to(self.device)
        new_state=new_state.to(self.device)
        action=action.to(self.device)
        reward=reward.to(self.device)
        done=done.to(self.device)
        
        #state=[preprocess_frame(frame, self.device) for frame in state] 
        #state=torch.cat(state)
        
        #new_state=[preprocess_frame(frame, self.device) for frame in new_state] 
        #new_state=torch.cat(new_state)
        
        #print('state batch shape {}'.format(state.shape))
        #print(state)
        
        #state=torch.Tensor(state).to(device)
        #new_state=torch.Tensor(new_state).to(device)
        
        
        reward=torch.Tensor(reward).to(self.device)
        
        #the view call below is to transform into column vector
        #so that it can be used in the gather call
        #i.e. we will use it to pick out from the computed value
        #tensor only values indexed by selected action
        action=(torch.Tensor(action).view(-1,1).long()).to(self.device)
        #print('action: ')
        #print(action)
        #print('contiguous?', action.is_contiguous())
        done=torch.Tensor(done).to(self.device)
        
        #print('shape of: state, new state, reward, action, done:')
        #print(state.shape)
        #print(new_state.shape)
        #print(reward.shape)
        #print(action.shape)
        #print(done.shape)
        
        
        self.nn.eval()
        self.target_nn.eval()
            
        with torch.no_grad():
            if self.config.double_dqn:
                #print('in double DQN')
                new_state_values_from_nn=self.nn(new_state).detach()
                #print('new_state_values_from_nn shape {} and value:'.format(new_state_values_from_nn.shape))
                #print(new_state_values_from_nn)
                max_new_state_indexes=torch.max(new_state_values_from_nn,dim=1)[1].view(-1,1)
                #print('max_new_state_indexes shape {} and value:'.format(max_new_state_indexes.shape))
                #print(max_new_state_indexes)
                new_state_values=self.target_nn(new_state).detach()
                #print('new_state_values shape {} and value:'.format(new_state_values.shape))
                #print(new_state_values)
                max_new_state_values=torch.gather(new_state_values,1,max_new_state_indexes).squeeze()
                #print('max_new_state_values shape {} and value:'.format(max_new_state_values.shape))
                #print(max_new_state_values)
            else:
                #print('in regular DQN')
                new_state_values=self.target_nn(new_state).detach()
                #print('new_state_values shape {} and value'.format(new_state_values.shape))
                #print(new_state_values)
            
                max_new_state_values=torch.max(new_state_values,dim=1)[0]
                #print('max_new_state_values shape {} and value'.format(max_new_state_values.shape))
                #print(max_new_state_values)
                
            target_value=(reward + (1-done)*self.config.gamma*max_new_state_values).view(-1,1)
        
        #end no grad
        
        #print('shape of: target_value')
        #print(target_value.shape)
        self.nn.train()
        
        #this will select only the values of the desired actions
        predicted_value=torch.gather(self.nn(state),1,action)
        #print('shape of: predicted_value')
        #print(predicted_value.shape)
        
        
        loss=self.loss_function(predicted_value,target_value)
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config.clip_error:
            for param in self.nn.parameters():
                param.grad.clamp_(-1.0,1.0)
        
        self.optimizer.step()
        
        if self.number_of_frames % self.config.update_target_frequency == 0:
            #print("***********************")
            #print("UPDATING TARGET NETWORK")
            #print("update counter: {}".format(self.update_target_counter))
            #print("***********************")
            self.target_nn.load_state_dict(self.nn.state_dict())
            
        if self.number_of_frames % self.config.save_model_frequency ==0:
            self.save_model()
        
        self.number_of_frames+=1