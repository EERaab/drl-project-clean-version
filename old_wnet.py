from settings import *
from features import *
from utils import *
from models import *

#For window net we do pointwise convolutions to condense information into a single layer.
#Because of the nature of the information (effectively answered by 2 yes/no questions and a 'how many'? we can compress the data into a 1D value which later layers, if they are well designed can seprate out.
#The there are effectively a few decision boundary thresholds that they have to distinguish
class Tile3LayerConvolutionOLD(nn.Module):
    def __init__(self):
        super(Tile3LayerConvolutionOLD, self).__init__()
        self.pw_conv1 = nn.Linear(3, 6) #We run 6 distinct pointwise convolutions over each point in the local windows
        self.pw_conv2 = nn.Linear(6, 1) #We map down each tile to a single value
            
    def forward(self, local_window_features):
        # input: B x W x H x 3
        xf = F.elu(self.pw_conv1(local_window_features)) #output B x W x H x 6
        xf = F.elu(self.pw_conv2(xf)).squeeze(-1) #output B x W x H
        return xf

#Reflection management for wnet
def window_directional_split(window_tensor, radius, stack_along_dim = 1):
    dir0 = window_tensor[:,radius+1:,:]
    dir1 = window_tensor[:,:,radius+1:].rot90(k=-1, dims=(1,2))
    dir2 = window_tensor[:,:radius,:].rot90(k=-2, dims=(1,2))
    dir3 = window_tensor[:,:,:radius].rot90(k=-3, dims=(1,2))
    #Stacking
    X = torch.stack((dir0,dir1,dir2,dir3), dim = stack_along_dim)
    #Reflection symmetrization (this is IDIOTIC since its giga-lossy, and should thus be removed! Reflection-symmetrization occurs in the end.)
    X = (torch.add(X,torch.flip(X, dims = (-1,)))/2.0)[:,:,:,:radius+1]  
    return X

class WindowNet(nn.Module):
    def __init__(self, window_radius = 3, dropout=0.2):
        super(WindowNet, self).__init__()
        self.radius = window_radius
        p = dropout

        self.pw_conv = Tile3LayerConvolutionOLD()
        self.flatten = nn.Flatten(start_dim=2)
        self.dense_layers = nn.Sequential(
                nn.Linear((self.radius**2) + self.radius, 3*(self.radius + 1)),
                nn.Dropout(p),
                nn.ELU(),
                nn.Linear(3*(self.radius + 1), (self.radius + 1)),
                nn.Dropout(p),
                nn.ELU(),
                nn.Linear((self.radius + 1), 1, bias=False)
            )
        
        #self.dense_1 = nn.Linear((self.radius**2) + self.radius, 3*(self.radius + 1))
        #self.dense_2 = nn.Linear(3*(self.radius + 1), (self.radius + 1))
        #self.dense_3 = nn.Linear((self.radius + 1), 1, bias=False)

    def forward(self, local_window_features):
        # input: B x W x H x 1
        xf = self.pw_conv(local_window_features)
        
        # Reflection symmetry enforcment and flattening. 
        df = window_directional_split(xf, self.radius) 

        df = self.flatten(df) #output B x D x (radius^2 + radius)
        
        #Linear layers to estimate how 'good' any given direction is
        #lf = F.elu(self.dense_1(df))
        #lf = F.elu(self.dense_2(lf))
        #lf = self.dense_3(lf).squeeze(-1) #output B x D 
        lf = self.dense_layers(df).squeeze(-1)
        return lf
        
    def internals (self, local_window_features):
        # input: B x W x H x 1
        xf = self.pw_conv(local_window_features)
        
        return xf.squeeze(-1)
        
#initialize features for 3-layer with 'visits' feature:
def visits_internal_state_init(state):
    internal_state = torch.cat(
        (
        torch.tensor(state.agent_map, dtype=torch.float).to(device), 
        torch.zeros((1, state.agent_map.shape[1], state.agent_map.shape[2]), dtype=torch.float).to(device)
        )
        , dim=0)
    return internal_state #i.e. 3-layer map

def visits_update_internal_state(state, internal_state, wradius = 2):
    for pt in circle_iterator_inbounds(state.position, wradius):
        internal_state[0, pt[0], pt[1]] = state.agent_map[0,pt[0], pt[1]]
        internal_state[1, pt[0], pt[1]] = state.agent_map[1,pt[0], pt[1]]
    internal_state[2, state.position[0],state.position[1]] += 1
    #internal_state[0:2, state.position[0]-2:state.position[0]+3, state.position[1]-2:state.position[1]+3] = torch.tensor(state.agent_map[:,state.position[0]-2:state.position[0]+3, state.position[1]-2:state.position[1]+3]).to(device)
    #internal_state[2, state.position[0], state.position[1]] += 1
    return internal_state

def window_slice(state, internal_state, wradius = 3): #this is dumb, it should be in-place
    out = (internal_state[:, state.position[0]-wradius:state.position[0]+1+wradius, state.position[1]-wradius:state.position[1]+1+wradius]).clone().transpose(0,1).transpose(1,2)
    out[:,:,2] = 1/(1+out[:,:,2])
    return out.unsqueeze(0)

wnet_tile_eval = lambda tile : wnet.pw_conv(tile)
wradius = 7


wnet_nn = WindowNet(window_radius = 3).to(device)
wnet_fm = FeatureMapping(
    lambda state, internal_state : window_slice(state, internal_state, 3),
    lambda state : visits_internal_state_init(state),
    lambda state, internal_state : visits_update_internal_state(state, internal_state)
)
wnet = ModelPair(wnet_nn, wnet_fm, 'wnet', False)
wnet_gen = lambda dropout = 0.2 : ModelPair(WindowNet(window_radius = 3, dropout = dropout).to(device), wnet_fm, 'wnet', False)

wnet_plus_nn = WindowNet(window_radius = 7).to(device)
wnet_plus_fm = FeatureMapping(
    lambda state, internal_state : window_slice(state, internal_state, 7),
    lambda state : visits_internal_state_init(state),
    lambda state, internal_state : visits_update_internal_state(state, internal_state)
)
wnet_plus = ModelPair(wnet_plus_nn, wnet_plus_fm, 'wnet_plus', False)
wnet_plus_gen = lambda dropout = 0.2 : ModelPair(WindowNet(window_radius = 7, dropout = dropout).to(device), wnet_plus_fm, 'wnet_plus', False)


class DeepWindowNet(nn.Module):
    def __init__(self, window_radius = 7, dropout_rate = 0.2, pw_conv = True):
        super(DeepWindowNet, self).__init__()
        self.radius = window_radius
        self.dropout_rate = dropout_rate
        if pw_conv:
            self.entry_layer = Tile3LayerConvolutionOLD()
        else:
            self.entry_layer = nn.Identity()
        R = window_radius
        p = dropout_rate

        self.flatten = nn.Flatten(start_dim=2)
        self.mid_layers = nn.Sequential(
            nn.Linear((R**2) + R, 4*((R**2) + R)),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear(4*((R**2) + R), 4*((R**2) + R)),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear(4*((R**2) + R), ((R**2) + R)),
        )

        self.later_layers = nn.Sequential(
            nn.Linear(((R**2) + R), 3*(R + 1)),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear(3*(R + 1), 3*(R + 1)),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear(3*(R + 1), (R + 1)),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear((R + 1), 1, bias=False)
        )

    def forward(self, local_window_features):
        # input: B x W x H x 1
        xf = self.entry_layer(local_window_features)
        
        xf = window_directional_split(xf, self.radius)
        xf = self.flatten(xf)
        yf = self.mid_layers(xf)
        xf = F.elu(torch.add(xf, yf)) #could remove skip connection here
        xf = self.later_layers(xf).squeeze(-1)
        return xf
        
    def internals(self, local_window_features):
        # input: B x W x H x 1
        xf = self.entry_layer(local_window_features).squeeze(-1)
        return xf
        

dwnet_nn = DeepWindowNet(window_radius = 7).to(device)
dwnet_fm = FeatureMapping(
    lambda state, internal_state : window_slice(state, internal_state, 7),
    lambda state : visits_internal_state_init(state),
    lambda state, internal_state : visits_update_internal_state(state, internal_state)
)
dwnet = ModelPair(dwnet_nn, dwnet_fm, 'dwnet', False)
dwnet_gen = lambda dropout = 0.2 : ModelPair(DeepWindowNet(window_radius = 7, dropout_rate = dropout).to(device), dwnet_fm, 'dwnet', False)


#Without the skip-connection of DWN. Doesn't appear to do much better in early tests.
class NoSkipDeepWindowNet(nn.Module):
    def __init__(self, window_radius = 7, dropout_rate = 0.2, pw_conv = True):
        super(NoSkipDeepWindowNet, self).__init__()
        self.radius = window_radius
        self.dropout_rate = dropout_rate
        if pw_conv:
            self.entry_layer = Tile3LayerConvolutionOLD()
        else:
            self.entry_layer = nn.Identity()
        R = window_radius
        p = dropout_rate

        self.flatten = nn.Flatten(start_dim=2)
        self.mid_layers = nn.Sequential(
            nn.Linear((R**2) + R, 4*((R**2) + R)),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear(4*((R**2) + R), 4*((R**2) + R)),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear(4*((R**2) + R), ((R**2) + R)),
        )

        self.later_layers = nn.Sequential(
            nn.Linear(((R**2) + R), 3*(R + 1)),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear(3*(R + 1), 3*(R + 1)),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear(3*(R + 1), (R + 1)),
            nn.ELU(),
            nn.Dropout(p),
            nn.Linear((R + 1), 1, bias=False)
        )

    def forward(self, local_window_features):
        # input: B x W x H x 1
        xf = self.entry_layer(local_window_features)
        
        xf = window_directional_split(xf, self.radius)
        xf = self.flatten(xf)
        yf = self.mid_layers(xf)
        xf = F.elu(yf) 
        xf = self.later_layers(xf).squeeze(-1)
        return xf
        
    def internals(self, local_window_features):
        # input: B x W x H x 1
        xf = self.entry_layer(local_window_features).squeeze(-1)
        return xf

noskipdwnet_nn = NoSkipDeepWindowNet(window_radius = 7).to(device)
noskipdwnet_fm = FeatureMapping(
    lambda state, internal_state : window_slice(state, internal_state, 7),
    lambda state : visits_internal_state_init(state),
    lambda state, internal_state : visits_update_internal_state(state, internal_state)
)
noskipdwnet = ModelPair(noskipdwnet_nn, noskipdwnet_fm, 'noskipdwnet', False)
noskipdwnet_gen = ModelPair(NoSkipDeepWindowNet(window_radius = 7).to(device), noskipdwnet_fm, 'noskipdwnet', False)
