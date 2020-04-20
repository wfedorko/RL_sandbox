import matplotlib.pyplot as plt

def plot_results(reward_total, i_episode,config):
    plt.figure(figsize=[12,5])
    plt.title("Score at end of episode")
    plt.plot(reward_total[:i_episode],color='red')
    plt.savefig('{}/{}/reward_histry.png'.format(config.result_path,config.exp_name))
    plt.close()