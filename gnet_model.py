from settings import *
from features import *
from utils import *
from models import *
from gnet_features import *
from symmetrization import *

DIAMETER = WORLD_BOUNDARY_PADDING*2 + 1 #this is the absolute maximum - we can make it smaller however. 


class GlobalConvolutionNet(nn.Module):
    def __init__(self, tile_size = GNET_TILE_SIZE, in_channels = 2, intermediate_out_channels = 1):
        super(GlobalConvolutionNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = tile_size, stride = tile_size, padding = 'valid'), 
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, intermediate_out_channels, kernel_size = 1, padding = 'valid'), #simplify pointwise data,
            nn.ELU()
        )
        #output B x out_layers x TH x TW

        self.positional_convolution = nn.Conv2d(1, 1, kernel_size = tile_size, stride = tile_size, padding = 'valid')
        self.positional_convolution.weight = nn.Parameter(torch.ones((1,1,tile_size,tile_size))/(tile_size**2), requires_grad=False) #this layer is frozen.
        #output B x 1 x TH x TW (could be done by fixing individual parts of a tensor)

        self.merge_layer = nn.Sequential(
            nn.Conv2d(intermediate_out_channels+1, 6, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(6, 1, kernel_size=1)
            )
    
    def forward(self, global_map):
        gmap_value = self.conv_layers(global_map)
        gmap_position = self.positional_convolution(global_map[:,0,:,:].unsqueeze(1))
        combined_conv = torch.cat((gmap_position, gmap_value), dim = 1)
        output = self.merge_layer(combined_conv)
        return output.squeeze(1)

class GlobalNet(nn.Module):
    def __init__(self, global_convolution, global_pathfinder, local_pathfinder, local_map_shape, directional = False):
        super(GlobalNet, self).__init__()

        self.pw_conv = Tile3LayerConvolution()
        self.global_conv = global_convolution
        self.global_pathfinder = global_pathfinder
        self.local_pathfinder = local_pathfinder
        self.local_map_shape = local_map_shape
        self.flatten = nn.Flatten(start_dim = -2)
        if directional:
            radius = (local_map_shape[0]-1)//2
            self.map_symmetrization = lambda amap : window_directional_split_with_reflection(amap, radius)
        else:
            self.map_symmetrization = map_symmetrizer
        

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
        #Input BxDxD, output Bx4
        adjustment_values = self.global_pathfinder(tile_map) 
        local_map = self.pw_conv(local_map).squeeze(1)

        #adjustment_values has shape Batch x Directions, local map has Batch x H x W and we set the 'frame values'
        #This is ugly, but there isn't much to do about it. It also BREAKS directional symmetry, but only slightly.
        local_map[:, 0, :] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[0])[:, 0, :]
        local_map[:, -1, :] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[0])[:, 2, :]
        local_map[:, :, 0] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[1])[:, 1, :]
        local_map[:, :, -1] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[1])[:, 3, :]

        #Finally we use a localized path finding algorithm to suggest an action value.

        sym_map = self.map_symmetrization(local_map)
        flat_sym_map = self.flatten(sym_map)
        sx = self.local_pathfinder(flat_sym_map)
        action_values = map_combiner(sx)
        
        return action_values
        

    def internals(self, local_map, global_map):
        with torch.no_grad():
            #To visualize the features while processing, before application of the local pathfinder.
            #We apply the condensing map taking our many channels and merging them into a single scalar for each point.
            #On the global map this is not applied to the position layer - we use a concatenation to handle this
            global_map_adj = self.pw_conv(global_map[:,1:,:,:])
            gmap = torch.cat((global_map[:,0,:,:].unsqueeze(1), global_map_adj), dim = 1)

            #The global map information is massively condensed into a way simplified map
            tile_map = self.global_conv(gmap)

            #To pad the local map (really, we replace tiles in the local map) we use a global pathfinder on the "tile map".
            #It should convey information about the direction the agent should go. 
            #It gives 4 values which are 'padded' to the local map.

            adjustment_values = self.global_pathfinder(tile_map) 
            lmap = self.pw_conv(local_map).squeeze(1)

            #adjustment_values has shape Batch x Directions, local map has Batch x H x W and we set the 'frame values'
            #This is ugly, but there isn't much to do about it. It also BREAKS directional symmetry!
            lmap[:, 0, :] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[0])[:, 0, :]
            lmap[:, :, 0] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[1])[:, 1, :]
            lmap[:, -1, :] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[0])[:, 2, :]
            lmap[:, :, -1] = adjustment_values.unsqueeze(-1).expand(-1, -1, self.local_map_shape[1])[:, 3, :]
        return lmap, tile_map

#gnet_nn = GlobalNet(GlobalConvolutionNet(), DensePathfinderNet(GLOBAL_TILE_NUMBER), DensePathfinderNet(DIAMETER), (DIAMETER,DIAMETER)).to(device)
gnet_fm = FeatureMapping(
    gnet_get_features,
    gnet_internal_state_init,
    gnet_update_internal_state
)
#gnet = ModelPair(gnet_nn, gnet_fm, 'gnet')
