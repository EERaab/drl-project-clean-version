from settings import *
from mdp import *
from utils import *
import random


#Constant generator
def const(value):
    while True:
        yield value

##### Replay buffer#####
#For SARSA we save the State S, the action A, the reward R, the next state S' and the next action A', hence the name.
#This is adjusted from https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Notably this *IS NOT* memory efficient due to the duplicates of all actions and states but the first and last.
# Thus we could look to optimize this if necessary.

#For one-feature ANNs:
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'next_action'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#For two-feature ANNs we use a different structure
MultiFeatTransition = namedtuple('MultiFeatTransition',
                        (
                        'primary_feat', 
                        'secondary_feat', 
                        'action', 
                        'reward', 
                        'next_primary_feat', 
                        'next_secondary_feat', 
                        'next_action'
                        )
                        )

class MultiFeatReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(MultiFeatTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def eps_greedy(epsilon, policy_choice, mark_eps=False):
    if random.random() < epsilon:
        #Since our action-range is 0,1,2,3:
        if mark_eps:
            return (torch.tensor(random.randrange(4), device = device) , 1)
        return torch.tensor(random.randrange(4), device = device)
    if mark_eps: 
        return (policy_choice, 0)
    return policy_choice
    
def softmax_choice(T):
    K = F.softmax(T, dim = 1)
    l = random.random()
    for i in range(4):
        if sum(K[0,:i+1]) > l:
            return torch.tensor(i).to(device) #stupid, but gotta be compatible with the default policy structure, which expects a tensor...
    return torch.tensor(3).to(device)

def softmax_policy(network_eval):
    return softmax_choice(network_eval)
    
def greedy_policy(network_eval):
    return network_eval.max(1).indices

def decay(value = 0.5, threshold = 0, decay_factor = 0.99):
    i = 0
    s = value*1.0
    while i  < threshold:
        yield s
        i+=1
    while True:
        s *= decay_factor
        yield s

### Training loop ###
def sarsa(
    inet,
    policy = False,
    world_generator = lambda : dense_world(), 
    learning_rate = LEARNING_RATE, 
    max_episodes=50, 
    episode_duration=50, 
    gamma = GAMMA, 
    epsilon=EPSILON, 
    threshold=np.inf, 
    decay_factor = 0.0
    ):

    network = inet.nn_model
    feature_map = inet.feature_mapping
    is_mf = inet.is_multi_feature

    if is_mf:
        eval_net = lambda network, feats : network(*feats)
        memory = MultiFeatReplayMemory(MEMORY_SIZE)
    else:
        eval_net = lambda network, feats : network(feats)
        memory = ReplayMemory(MEMORY_SIZE)
    
    if not policy:
        policy = lambda feats, network, eps : eps_greedy(eps, greedy_policy(eval_net(network, feats)))
    rewards = []
    
    trainable_params = [param for param in network.parameters() if param.requires_grad] #might be a bit slow    
    optimizer = optim.AdamW(trainable_params, lr = learning_rate, amsgrad=True)
    
    epsilon_gen = decay(epsilon, threshold, decay_factor)

    for i in range(max_episodes):
        episode_total_reward = 0
        world = world_generator()
        state = initialize_state(world)
        epsilon = next(epsilon_gen)
        internal_state = feature_map.initialize_internal_state(state)

        S  = feature_map.get_features(state, internal_state)
        A  = policy(S, network, epsilon)
        R  = torch.tensor([state.transition(A.item(), world)], device= device)
        S_ = feature_map.get_features(state, internal_state)
        A_ = policy(S_, network, epsilon)

        if is_mf:
            memory.push(S[0], S[1], A, R, S_[0], S_[1], A_)
        else:
            memory.push(S, A, R, S_, A_)
        episode_total_reward += R.item()
        
        for j in range(episode_duration):
            S  = S_
            A  = A_
            R  = torch.tensor([state.transition(A.item(), world)], device= device)
            if j < episode_duration-1:                
                S_ = feature_map.get_features(state, internal_state)
                A_ = policy(S_, network, epsilon)
                if is_mf:
                    memory.push(S[0], S[1], A, R, S_[0], S_[1], A_)
                    optimize_model_mf(network, optimizer, memory, gamma)
                else:
                    memory.push(S, A, R, S_, A_)
                    optimize_model(network, optimizer, memory, gamma)
            
            episode_total_reward += R.item()
        rewards.append(episode_total_reward)
    return rewards


#We lean on the DQN implementation on https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#to see how to create the optimizer etc.
def optimize_model(network, optimizer, memory, gamma):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.

    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    next_state_batch =  torch.cat(batch.next_state)
    action_batch = torch.tensor(batch.action, device=device).unsqueeze(-1)
    reward_batch = torch.tensor(batch.reward, device=device).unsqueeze(-1)
    next_action_batch = torch.tensor(batch.next_action, device=device).unsqueeze(-1)

     # Compute Q(s, a), Q(s', a').
    state_action_values = network(state_batch).gather(1, action_batch)
    with torch.no_grad():
        next_state_action_values = network(next_state_batch).gather(1, next_action_batch)

    #We compute the TD-target.
    expected_state_action_values = (next_state_action_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(network.parameters(), 100)
    optimizer.step()

def optimize_model_mf(network, optimizer, memory, gamma):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    #For multi-feats we use
    #('primary_feat', 'secondary_feat', 'action', 'reward', 'next_primary_feat', 'next_secondary_feat' 'next_action')
    batch = MultiFeatTransition(*zip(*transitions))
    primary_feat_batch = torch.cat(batch.primary_feat)
    secondary_feat_batch = torch.cat(batch.secondary_feat)
    next_primary_feat_batch =  torch.cat(batch.next_primary_feat)
    next_secondary_feat_batch = torch.cat(batch.next_secondary_feat)
    action_batch = torch.tensor(batch.action, device=device).unsqueeze(-1)
    reward_batch = torch.tensor(batch.reward, device=device).unsqueeze(-1)
    next_action_batch = torch.tensor(batch.next_action, device=device).unsqueeze(-1)

     # Compute Q(s, a), Q(s', a').
    state_action_values = network(primary_feat_batch, secondary_feat_batch).gather(1, action_batch)
    with torch.no_grad():
        next_state_action_values = network(next_primary_feat_batch, next_secondary_feat_batch).gather(1, next_action_batch)

    #We compute the TD-target.
    expected_state_action_values = (next_state_action_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(network.parameters(), 100)
    optimizer.step()



## We shall seek to plot full episodes and the inner workings of our network.
Outcome = namedtuple("Outcome",('states', 'assesments', 'rewards', 'internal_states', 'features', 'epsilon_events', 'intermediates'))

def run_episode(
    world,
    inet,
    policy = False,
    max_episodes=50, 
    episode_duration=50, 
    epsilon=EPSILON, 
    threshold=np.inf, 
    decay_factor = 0.0, 
    ):
    
    network = inet.nn_model
    feature_map = inet.feature_mapping
    is_mf = inet.is_multi_feature
    
    network.eval()
    with torch.no_grad():    
        #this copy-paste from sarsa is a bit sloppy - does the torch.no_grad() apply?
        if is_mf:
            eval_net = lambda net, feats : net(*feats)
            eval_int = lambda net, feats : net.internals(*feats)
            memory = MultiFeatReplayMemory(MEMORY_SIZE)
        else:
            eval_net = lambda net, feats : net(feats)
            eval_int = lambda net, feats : net.internals(feats)
            memory = ReplayMemory(MEMORY_SIZE)
        
        if not policy:
            policy = lambda feats, net, eps : eps_greedy(eps, greedy_policy(eval_net(net, feats)), mark_eps = True)
        
        rewards = []
        states = []
        internal_states = []
        assesments = []
        feats = []
        epsilon_events = []
        intermediates = []
    
        state = initialize_state(world)    
        states.append(state.copy())
        internal_state = feature_map.initialize_internal_state(state)
        internal_states.append(internal_state.clone())
    
        S  = feature_map.get_features(state, internal_state)
        if is_mf:
            feats.append(S[0].clone())
        else:
            feats.append(S.clone())
        A, i  = policy(S, network, epsilon)
        intermediates.append(eval_int(network,S))
        assesments.append([round(float(x),3) for x in eval_net(network, S).squeeze().detach().cpu().numpy()])
        epsilon_events.append(i)
        R  = torch.tensor([state.transition(A.item(), world)], device= device) #goofy
        S_ = feature_map.get_features(state, internal_state)
        A_, i_ = policy(S_, network, epsilon)
        rewards.append(R.item())
    
        for j in range(episode_duration): #technically this means the episode duration is +1 relative to what we input...
            states.append(state.copy())
            internal_states.append(internal_state.clone())
            assesments.append([round(float(x),3) for x in eval_net(network, S_).squeeze().detach().cpu().numpy()])
            epsilon_events.append(i_)
            S  = S_
            intermediates.append(eval_int(network,S))
            if is_mf:
                feats.append(S[0].clone())
            else:
                feats.append(S.clone())
            A  = A_
            R  = torch.tensor([state.transition(A.item(), world)], device= device) #goofy
            if j < episode_duration-1:                
                S_ = feature_map.get_features(state, internal_state)
                A_, i_ = policy(S_, network, epsilon)
                
            rewards.append(R.item())
    network.train()
    return Outcome(states, assesments, rewards, internal_states, feats, epsilon_events, intermediates)



#plotting for wnet and others
def plot_aug_state(state, local_map, feature, world_map, rewards, assesment = None, pause = 0.5, i=0, radius = 3, eps = [], eps_ind = [], intermediate = None, is_wnet=False):
    plt.clf()
    lis = local_map.detach().cpu().numpy()
    feat = feature.detach().cpu().numpy()[0,:,:,:]

    if intermediate is not None:
        fig, ax = plt.subplots(ncols=4, nrows = 3, figsize = (15, 10))        
    else:
        fig, ax = plt.subplots(ncols=4, nrows = 2, figsize = (15, 10))
        
    if assesment:
        if eps[i]:
            fig.suptitle(f"Assesment: {assesment} (overridden by epsilon-greedy action) \n in the order Up, Right, Down, Left")
        else:
            fig.suptitle(f"Assesment: {assesment} \n in the order Up, Right, Down, Left")
    if rewards:
        ax[0,0].plot(rewards)
        ax[0,0].set_title('Episode rewards per step \n Current step in blue \n Epsilon events in red')
        ax[0,0].axvline(x = i, color = 'b', ls = '-')
        for j in eps_ind:
            ax[0,0].scatter(x=j, y=0, color="red", clip_on=False, marker = 2, transform=ax[0,0].get_xaxis_transform())
    else:
        fig.delaxes(ax[0][0])
    
    #the state
    fmap = state.flatten_map()
    swindow = window(state.flatten_map(), state, radius)
    
    colors_list = ['#33cc33', '#0099ff']
    masked_array = np.ma.masked_where(swindow < 0 , swindow)
    cmap = colors.ListedColormap(colors_list)
    cmap.set_bad(color = "black")
    ax[0,1].imshow(masked_array, cmap=cmap, origin="lower")
    ax[0,1].plot([radius], [radius], '-o', color = '#fff200')
    ax[0,1].axis('off')
    ax[0,1].set_title('Agent map')

    #the internals ~ the state, apart from layer 3 which we display
    is_window = window(lis, state, radius)
    d1_plot = ax[0,2].imshow(is_window, cmap="PuRd", origin="lower") 
    ax[0,2].plot([radius], [radius], '-o', color = '#fff200')
    ax[0,2].axis('off')
    ax[0,2].set_title('Internal "visits" state')
    plt.colorbar(d1_plot,ax=ax[0,2])

    #the agents full world map    
    masked_array2 = np.ma.masked_where(fmap < 0 , fmap)
    ax[0,3].imshow(masked_array2, cmap=cmap, origin="lower")
    ax[0,3].plot(state.position[1], state.position[0], '-ro', color = '#fff200')
    ax[0,3].axis('off')
    ax[0,3].set_title('Agent full world map')

    if is_wnet:
        #the features
        d2_plot = ax[1,0].imshow(feat[:,:,0], cmap="GnBu",origin="lower")
        ax[1,0].plot([radius], [radius], '-h', color = '#fff200')
        ax[1,0].axis('off')
        ax[1,0].set_title('"Seen" feature')
        plt.colorbar(d2_plot,ax=ax[1,0])
        
        d3_plot = ax[1,1].imshow(feat[:,:,1], cmap="BuPu_r", origin="lower")
        ax[1,1].plot([radius], [radius], '-h', color = '#fff200')
        ax[1,1].axis('off')
        ax[1,1].set_title('"Terrain" feature')
        plt.colorbar(d3_plot,ax=ax[1,1])
        
        d4_plot = ax[1,2].imshow(feat[:,:,2], cmap="BuGn_r", origin="lower")
        ax[1,2].plot([radius], [radius], '-h', color = '#fff200')
        ax[1,2].axis('off')
        ax[1,2].set_title('"Visits" feature')
        plt.colorbar(d4_plot,ax=ax[1,2])
    else:
        #the features
        d2_plot = ax[1,0].imshow(feat[0,:,:], cmap="GnBu",origin="lower")
        ax[1,0].plot([radius], [radius], '-h', color = '#fff200')
        ax[1,0].axis('off')
        ax[1,0].set_title('"Seen" feature')
        plt.colorbar(d2_plot,ax=ax[1,0])
        
        d3_plot = ax[1,1].imshow(feat[1,:,:], cmap="BuPu_r", origin="lower")
        ax[1,1].plot([radius], [radius], '-h', color = '#fff200')
        ax[1,1].axis('off')
        ax[1,1].set_title('"Terrain" feature')
        plt.colorbar(d3_plot,ax=ax[1,1])
        
        d4_plot = ax[1,2].imshow(feat[2,:,:], cmap="BuGn_r", origin="lower")
        ax[1,2].plot([radius], [radius], '-h', color = '#fff200')
        ax[1,2].axis('off')
        ax[1,2].set_title('"Visits" feature')
        plt.colorbar(d4_plot,ax=ax[1,2])
        

    #full world map
    ax[1,3].imshow(world_map, cmap=cmap, origin="lower")
    ax[1,3].plot(state.position[1], state.position[0], '-o', color = '#fff200')
    ax[1,3].axis('off')
    ax[1,3].set_title('Actual full world map')
    
    if intermediate is not None:
        if type(intermediate) == tuple:
            #simplified world map (ONLY WORKS IF TILE MAP IS DIVIDED IN 7x7 BLOCKS)
            gm = ax[2,3].imshow(intermediate[1][0,:,:].detach().cpu().numpy(), cmap='viridis', origin="lower")
            ax[2,3].plot(state.position[1]//7, state.position[0]//7, '-o', color = '#000000')
            ax[2,3].axis('off')
            ax[2,3].set_title('Simplified full world map')
            plt.colorbar(gm,ax=ax[2,3])
    
            #local map after adjustment
            lmap_plt = ax[2,1].imshow(intermediate[0][0,:,:].detach().cpu().numpy(), cmap='viridis', origin="lower")
            ax[2,1].plot(radius, radius, '-o', color = '#000000')
            ax[2,1].axis('off')
            ax[2,1].set_title('Adjusted local map')
            plt.colorbar(lmap_plt,ax=ax[2,1])
            
            fig.delaxes(ax[2,2])
            fig.delaxes(ax[2,0])
        else:
            #local map after adjustment
            lmap_plt = ax[2,1].imshow(intermediate[0,:,:].detach().cpu().numpy(), cmap='viridis', origin="lower")
            ax[2,1].plot(radius, radius, '-o', color = '#000000')
            ax[2,1].axis('off')
            ax[2,1].set_title('Adjusted local map')
            plt.colorbar(lmap_plt,ax=ax[2,1])
            
            fig.delaxes(ax[2,3])
            fig.delaxes(ax[2,2])
            fig.delaxes(ax[2,0])
    plt.close()    
    return fig
    
def plot_aug_episode(outcome, world_map, pause = 0.1, radius = 7, is_wnet=False):
    plt.ioff()
    states = outcome.states
    rewards = outcome.rewards
    assesments = outcome.assesments
    internal_states = outcome.internal_states
    features = outcome.features
    epsilon_events = outcome.epsilon_events
    intermediates = outcome.intermediates

    eps_indices = [i for i in range(len(epsilon_events)) if epsilon_events[i] > 0]
    for i in range(len(states)):
        if is_wnet:
            local_map = internal_states[i][2,:,:]
        else:
            local_map = internal_states[i][3,:,:]
            
        fig=plot_aug_state(states[i], local_map, features[i], world_map, rewards, assesments[i], i=i, eps = epsilon_events , eps_ind =eps_indices, radius = radius, intermediate = intermediates[i], is_wnet=is_wnet)
        display.display(fig)        
        plt.pause(pause)
        display.clear_output(wait=True)
    plt.ion()