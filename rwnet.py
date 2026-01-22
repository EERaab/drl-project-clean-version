from settings import *
from features import *
from utils import *
from models import *
from gnet_features import *
from gnet_model import *
from symmetrization import *

#were a bit lazy: lnet uses the multi-feature methods though it is wholly independent of the global part.

class LocalNet(nn.Module):
    def __init__(self, local_pathfinder, radius):
        super(LocalNet, self).__init__()

        self.pw_conv = Tile3LayerConvolution()
        self.local_pathfinder = local_pathfinder
        self.flatten = nn.Flatten(start_dim = 3)
        self.radius = radius
        
    def forward(self, local_map, global_map):
        #In-shape BxCxHxW, out-shape: BxHxW
        lmap = self.pw_conv(local_map).squeeze(1)
        #In shape BxHxW, out shape: BxDx2xHxR 
        sym_map = window_directional_split_with_reflection(lmap, self.radius)
        #In shape: BxDx2xHxR, out shape: BxDx2xHxR
        flat_sym_map = self.flatten(sym_map)        
        #In shape: BxDxRxHW, out shape: BxDxR
        symd = self.local_pathfinder(flat_sym_map)   
        #In shape: BxDxR, out shape: BxD
        action_values = map_combiner(symd)
        return action_values
        
    def internals(self, local_map, global_map):
        #In-shape BxCxHxW, out-shape: BxHxW
        lmap = self.pw_conv(local_map).squeeze(1)
        return lmap

class SimpleLocalPathfinderNet(nn.Module):
    def __init__(self, radius):
        super(SimpleLocalPathfinderNet, self).__init__()
                
        self.layers = nn.Sequential(
                nn.Linear(radius*(2*radius+1), 3*(2*radius+1)),
                nn.ELU(),
                nn.Linear( 3*(2*radius+1), (2*radius+1)),
                nn.ELU(),
                nn.Linear((2*radius+1), 1)
            )
    
    def forward(self, amap):
        x = self.layers(amap).squeeze(-1)
        return x


rwnet_nn = LocalNet(SimpleLocalPathfinderNet(7), radius = 7).to(device)
rwnet_fm = gnet_fm
rwnet = ModelPair(rwnet_nn, rwnet_fm, 'rwnet', True)
rwnet_gen = lambda : ModelPair(LocalNet(SimpleLocalPathfinderNet(7), radius = 7).to(device), rwnet_fm, 'rwnet', True)






class GlobalNet(nn.Module):
    def __init__(self, global_convolution, global_pathfinder, local_pathfinder, local_map_shape, directional = False, flatten_dim = 3):
        super(GlobalNet, self).__init__()

        self.pw_conv = Tile3LayerConvolution()
        self.global_conv = global_convolution
        self.global_pathfinder = global_pathfinder
        self.local_pathfinder = local_pathfinder
        self.local_map_shape = local_map_shape
        if directional:
            radius = (local_map_shape[0]-1)//2
            self.map_symmetrization = lambda amap : window_directional_split_with_reflection(amap, radius)
        else:
            self.map_symmetrization = map_symmetrizer
        self.flatten = nn.Flatten(start_dim=flatten_dim)

    def forward(self, local_map, global_map):
        #We apply the condensing map taking our many channels and merging them into a single scalar for each point.
        #On the global map this is not applied to the position layer - we use a concatenation to handle this
        global_map_adj = self.pw_conv(global_map[:,1:,:,:])
        global_map = torch.cat((global_map[:,0,:,:].unsqueeze(1), global_map_adj), dim = 1)

        #The global map information is massively condensed into a way simplified map
        tile_map = self.global_conv(global_map)

        #To pad the local map (really, we replace tiles in the local map) we use a global pathfinder on the "tile map".
        #It should convey information about the direction the agent should go. 
        #It gives 4 values which are 'padded' to the local map.

        adjustment_values = self.global_pathfinder(tile_map) 

        
        local_map = self.pw_conv(local_map).squeeze(1)

        #adjustment_values has shape Batch x Directions, local map has Batch x H x W and we set the 'frame values'
        #This is ugly, but there isn't much to do about it. It also BREAKS directional symmetry!
        local_map[:, 0, :] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[0])[:, 0, :]
        local_map[:, -1, :] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[0])[:, 2, :]
        local_map[:, :, 0] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[1])[:, 1, :]
        local_map[:, :, -1] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[1])[:, 3, :]

        #Finally we use a localized path finding algorithm to suggest an action value.
        sym_map = self.map_symmetrization(local_map)
        sx=self.flatten(sym_map)
        sx = self.local_pathfinder(sx)
        action_values = map_combiner(sx)
        
        return action_values

    def internals(self, local_map, global_map):
        #We apply the condensing map taking our many channels and merging them into a single scalar for each point.
        #On the global map this is not applied to the position layer - we use a concatenation to handle this
        global_map_adj = self.pw_conv(global_map[:,1:,:,:])
        global_map = torch.cat((global_map[:,0,:,:].unsqueeze(1), global_map_adj), dim = 1)

        #The global map information is massively condensed into a way simplified map
        tile_map = self.global_conv(global_map)

        #To pad the local map (really, we replace tiles in the local map) we use a global pathfinder on the "tile map".
        #It should convey information about the direction the agent should go. 
        #It gives 4 values which are 'padded' to the local map.

        adjustment_values = self.global_pathfinder(tile_map) 

        
        local_map = self.pw_conv(local_map).squeeze(1)

        #adjustment_values has shape Batch x Directions, local map has Batch x H x W and we set the 'frame values'
        #This is ugly, but there isn't much to do about it. It also BREAKS directional symmetry!
        local_map[:, 0, :] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[0])[:, 0, :]
        local_map[:, -1, :] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[0])[:, 2, :]
        local_map[:, :, 0] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[1])[:, 1, :]
        local_map[:, :, -1] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[1])[:, 3, :]
        
        return (local_map, tile_map)

        
        
class SimpleGlobalPathfinderNet(nn.Module):
    def __init__(self, radius, flatten_dim_start = 1):
        super(SimpleGlobalPathfinderNet, self).__init__()
                
        self.layers = nn.Sequential(
                nn.Flatten(start_dim=flatten_dim_start),
                nn.Linear((2*radius+1)**2, 6*(2*radius+1)),
                nn.ELU(),
                nn.Linear(6*(2*radius+1), 2*(2*radius+1)),
                nn.ELU(),
                nn.Linear(2*(2*radius+1), 4)
            )
    
    def forward(self, amap):
        x = self.layers(amap).squeeze(-1)
        return x
        
grwnet_nn = GlobalNet(GlobalConvolutionNet(), SimpleGlobalPathfinderNet(4), SimpleLocalPathfinderNet(7), (15,15), directional =True ).to(device)
grwnet_fm = FeatureMapping(
    gnet_get_features,
    gnet_internal_state_init,
    gnet_update_internal_state
)
grwnet = ModelPair(grwnet_nn, grwnet_fm, 'grwnet', True)
grwnet_gen = lambda : ModelPair(GlobalNet(GlobalConvolutionNet(), SimpleGlobalPathfinderNet(4), SimpleLocalPathfinderNet(7), (15,15), directional =True).to(device), grwnet_fm, 'grwnet', True)