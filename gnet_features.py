from settings import * 
from utils import * 

GNET_TILE_SIZE = 7 

#initialize features for 4-layer with 'visits' feature:
def gnet_internal_state_init(state):
    internal_state = torch.cat(
        (
        torch.zeros((1, state.agent_map.shape[1], state.agent_map.shape[2]), dtype=torch.float).to(device),     #POSITIONAL ENCODING
        torch.tensor(state.agent_map, dtype=torch.float).to(device),                                            #AGENT MAP
        torch.zeros((1, state.agent_map.shape[1], state.agent_map.shape[2]), dtype=torch.float).to(device)      #VISITS
        )
        , dim=0)
    x = (state.position[0]//GNET_TILE_SIZE)*GNET_TILE_SIZE
    y = (state.position[1]//GNET_TILE_SIZE)*GNET_TILE_SIZE
    internal_state[0, x:(x+GNET_TILE_SIZE), y:(y+GNET_TILE_SIZE)] += 1 #(or we could just take a single entry)
    return internal_state #i.e. 4-layer map

def gnet_update_internal_state(state, internal_state, wradius = 2):
    for pt in circle_iterator_inbounds(state.position, wradius):
        internal_state[1, pt[0], pt[1]] = state.agent_map[0,pt[0], pt[1]]
        internal_state[2, pt[0], pt[1]] = state.agent_map[1,pt[0], pt[1]]

    internal_state[3, state.position[0],state.position[1]] += 1

    x = (state.position[0]//GNET_TILE_SIZE)*GNET_TILE_SIZE
    y = (state.position[1]//GNET_TILE_SIZE)*GNET_TILE_SIZE
    if not internal_state[0,x,y] == 1:
        internal_state[0,:,:] = 0
        internal_state[0,x:x+GNET_TILE_SIZE,y:y+GNET_TILE_SIZE] += 1
    return internal_state

def gnet_get_features(state, internal_state, wradius = 7): #this is dumb, it should be in-place
    local_map = (internal_state[1:, state.position[0]-wradius:state.position[0]+1+wradius, state.position[1]-wradius:state.position[1]+1+wradius]).clone()
    local_map[2,:,:] = 1/(1+local_map[2,:,:])
    local_map = local_map.unsqueeze(0)
    global_map = internal_state.clone()
    global_map = global_map.unsqueeze(0)
    return local_map, global_map