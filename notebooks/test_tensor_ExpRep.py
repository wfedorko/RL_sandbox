import sys  
sys.path.insert(0, '../')
sys.path.insert(0, '../baselines')

import matplotlib.pyplot as plt

from datetime import datetime

import random


class CONFIG:
    pass

config=CONFIG()

config.use_cuda=True
config.device="cuda:5"

config.env_id="PongNoFrameskip-v4"
config.exp_name=config.env_id+"-"+datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
#config.result_path = get_ipython().run_line_magic('pwd', '')
config.result_path= '/home/wfedorko/RL_sandbox/notebooks/../results/'
print('result path is'+config.result_path)

config.learning_rate = 0.0001
config.num_episodes = 1000
config.gamma=0.99
#gamma=0.85
config.egreedy = 0.9
config.egreedy_final = 0.01
config.egreedy_decay = 10000

config.report_interval=10

config.score_to_solve = 18.0

config.hidden_layer_size=512

#config.replay_memory_size=100000
config.replay_memory_size=100

config.batch_size=32

config.update_target_frequency = 5000

config.clip_error=True
config.normalize_image=True

config.double_dqn=True

config.save_movies=True

config.save_model_frequency=10000

config.resume_previous_training=False

#config.memory_device=config.device
config.memory_device='cpu'


from utils.engine import Engine

my_engine=Engine(config)

# get_ipython().run_line_magic('load_ext', 'line_profiler')

my_engine.train_agent()





