from settings import *
import pandas as pd
import scipy as sp

# This is ripped from my project 5 report.
def run_training(network_gen, training_function, test_function, sessions = 3):    
    rewards_df = pd.DataFrame() 
    test_rewards_df = pd.DataFrame() 
    for sesh_nr in range(sessions):
        sesh_str = 'Session ' + str(sesh_nr)
        #Create a new agent
        net = network_gen()
        #Train it
        rewards = training_function(net)
        rewards_df[sesh_str] = rewards        
        #Test it (eval mode: No dropout etc)
        test_rewards = test_function(net)
        test_rewards_df[sesh_str] = test_rewards
        del net
    return rewards_df, test_rewards_df
        
def roll_training_data(rewards_df, roll_nr=5):
    rolled_df = rewards_df.rolling(roll_nr).sum()/roll_nr
    rolled_df['Mean'] = rolled_df.mean(axis = 1)
    rolled_df['Err.'] = rolled_df.agg(sp.stats.sem, axis = 1)
    return rolled_df

def plot_training(rolled_df, episodes, sessions = 3, roll_nr=5, error_bar_freq = False):
    if not error_bar_freq:
        error_bar_freq = max(1,int(episodes/50))
    eb = plt.errorbar(range(1, episodes+1), rolled_df['Mean'], yerr = rolled_df['Err.'], errorevery = error_bar_freq, elinewidth=1.5, capsize=1.5, barsabove=False, ls='-', lw = 1.5)
    plt.xlabel(f'Episode number (rolled over {roll_nr})', fontsize = 10)
    eb[-1][0].set_linestyle('--')
    plt.ylabel(f'Rolling rewards averaged across {sessions} sessions.', fontsize = 10)
    plt.title("Training performance")
    return eb

def multiplot_training(rolled_dfs, names, episodes, sessions=5, roll_nr=False, error_bar_freq = False, ep_min = 1, ep_max = False):
    if ep_max:
        episodes = ep_max
    if not error_bar_freq:
        error_bar_freq = max(1,int(episodes/50))
    if not roll_nr:
        roll_nr = max(1,int(episodes/50))
    i=0
    fig = plt.figure()
    for rolled_df in rolled_dfs:
        plt.errorbar(range(ep_min, episodes+1), rolled_df['Mean'][ep_min-1:episodes], yerr = rolled_df['Err.'][ep_min-1:episodes], 
            errorevery = error_bar_freq, elinewidth=1.5, capsize=1.5, barsabove=False, lw = 1.5, label=names[i])
        i+=1
    plt.xlabel(f'Episode number (rolled over {roll_nr})', fontsize = 10)
    plt.ylabel(f'Rolling rewards averaged across {sessions} sessions.', fontsize = 10)
    plt.title("Training performance")
    plt.legend(loc=4)
    #plt.show()
    return fig


def metrics(test_df):
    test_summary_df = pd.DataFrame() 
    test_summary_df['Mean'] = test_df.mean(axis=0)
    test_summary_df['Err.'] = test_df.sem(axis=0)
    print(test_summary_df.head())
    best_sesh = test_summary_df['Mean'].idxmax()
    mean = test_summary_df['Mean'][best_sesh]
    err = test_summary_df['Err.'][best_sesh]
    return best_sesh, mean, err

def plot_test(test_df, perf):
    scores = test_df[perf]
    plt.plot(scores)
    plt.xlabel('World nr.', fontsize = 10)    
    plt.ylabel('Reward.', fontsize = 10)
    plt.title('Test performance across 50 worlds')
    return None