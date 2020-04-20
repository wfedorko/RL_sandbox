import math

def calculate_epsilon(steps_done,config):
    epsilon = config.egreedy_final + (config.egreedy - config.egreedy_final) * \
              math.exp(-1. * steps_done / config.egreedy_decay )
    return epsilon

