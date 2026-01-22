from settings import *
from features import *
from models import *

class SymmetrizationOfTinyLinear(nn.Module):
    #To take a 16-vector X and map it into a 4-vector A such that A_i = f(X[i:i+4]) and f(x,y,z,å) = f(x,y,å,z)
    # we can use a linear 4x1 on a Bx4x4 tensor if we force the linear to observe the last symmetry 
    def __init__(self):
        super(SymmetrizationOfTinyLinear, self).__init__()
        self.symmetrizer = torch.tensor([[1,0,0,0], [0,1,0,0], [0,0,1,1],[0,0,1,1]], device=device, dtype = torch.float)

    def forward(self, X):
        row = torch.matmul(X,self.symmetrizer)
        return row

class CombiNet(nn.Module):
    def __init__(self):
        super(CombiNet, self).__init__()
        # We asses each tile by the same metric, judging whether it is 'good' or bad by some linear + relu estimation combining the weighted visits and accessibility
        self.dense1 = nn.Linear(2, 1) #basically a pointwise conv across 2 channels.

        #We ensure reflection symmetry by forcing it here. Rotation symmetry follows from the features splitting into dimensions across which we apply the *same* linear transf.
        self.dense2 = parametrize.register_parametrization(nn.Linear(4, 1).to(device), "weight", SymmetrizationOfTinyLinear())
        
    def forward(self, tf):
        tf = F.elu(self.dense1(tf)).squeeze(-1) #combines pointwise the values of all tiles (visits and terrain type!)
        tf = self.dense2(tf).squeeze(-1)
        return tf

    def internals(self, tf):
        with torch.no_grad():
            lmap = F.elu(self.dense1(tf)).squeeze(-1)
        return lmap

cnet_nn = CombiNet().to(device)

def tiny_map_features(position, amap, dim = 1):
    #Not the most compact way, but it works
    d0_feats = torch.tensor((
                amap[dim,position[0]+1,position[1]],
                amap[dim,position[0]+2,position[1]],
                amap[dim,position[0]+1,position[1]+1],
                amap[dim,position[0]+1,position[1]-1]
                ), device=device, dtype = torch.float)
    d1_feats = torch.tensor((
                amap[dim,position[0],position[1]+1],
                amap[dim,position[0],position[1]+2],
                amap[dim,position[0]-1,position[1]+1],
                amap[dim,position[0]+1,position[1]+1]
                ), device=device, dtype = torch.float)
    d2_feats = torch.tensor((
                amap[dim,position[0]-1,position[1]],
                amap[dim,position[0]-2,position[1]],
                amap[dim,position[0]-1,position[1]-1],
                amap[dim,position[0]-1,position[1]+1]
                ), device=device, dtype = torch.float)
    d3_feats = torch.tensor((
                amap[dim,position[0],position[1]-1],
                amap[dim,position[0],position[1]-2],
                amap[dim,position[0]+1,position[1]-1],
                amap[dim,position[0]-1,position[1]-1]
                ), device=device, dtype = torch.float)
    return torch.stack((d0_feats, d1_feats, d2_feats, d3_feats), dim = -2).unsqueeze(0)

def visual_field_features(state):
    return tiny_map_features(state.position, state.agent_map)

def visits_ini(state):
    return torch.zeros((state.agent_map.shape[1],state.agent_map.shape[2]), device=device).unsqueeze(0)

def visits_upd(state, feature_state):
    feature_state[0,state.position[0], state.position[1]] += 1
    return feature_state

def weighted_mixed_features(state, fst):
    visual_features = visual_field_features(state)
    visits_features = tiny_map_features(state.position, fst, dim = 0)
    #We cannot judge tiles that we cannot visit by their number of visits. Instead we set these tiles to 0 and all other tiles to 1/(1+number) of visits.
    #Thus a number close to zero means we should avoid the tile.
    weighted_visits = (1-visual_features)/(1+visits_features) 
    return torch.stack((visual_features, weighted_visits), dim = -1)

cnet_fm = FeatureMapping(
    weighted_mixed_features,
    lambda st : visits_ini(st),
    visits_upd
    )

cnet = ModelPair(cnet_nn, cnet_fm, 'cnet', False)
gen_cnet = lambda : ModelPair(CombiNet().to(device), cnet_fm, 'cnet', False)