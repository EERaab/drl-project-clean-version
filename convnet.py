from settings import *
from features import *
from utils import *
from models import *
from gnet_features import *
from gnet_model import *
from symmetrization import *

#This is effectively the same as LocalNet (used for rwnet) but without the flatting first. (Could merge the two if I moved flattening, but the reshaping is goofy) 
class LocalConvNet(nn.Module):
    def __init__(self, local_conv, radius):
        super(LocalConvNet, self).__init__()

        self.pw_conv = Tile3LayerConvolution()
        self.local_pathfinder = local_conv
        self.radius = radius
        
    def forward(self, local_map, global_map):
        #In-shape BxCxHxW, out-shape: BxHxW
        lmap = self.pw_conv(local_map).squeeze(1)
        #In shape BxHxW, out shape: Bx4x2xHxR 
        sym_map = window_directional_split_with_reflection(lmap, self.radius)
        #Reshape into 8B x 1 x H x R for convolutions. Directionalization is a bit problematic here.
        B=sym_map.shape[0]
        reshaped_sym_map = sym_map.reshape (4*B, 2, 2*self.radius+1, self.radius)
        reshaped_sym_map = reshaped_sym_map.reshape (8*B, 1, 2*self.radius+1, self.radius)
        #In shape: 8BxHxR, out shape: 8B
        symd = self.local_pathfinder(reshaped_sym_map).squeeze(-1)
        symd = symd.reshape(4*B, 2)
        symd = symd.reshape(B, 4, 2)
        #In shape: Bx4x2, out shape: Bx4
        action_values = map_combiner(symd)
        return action_values
        
    def internals(self, local_map, global_map):
        #In-shape BxCxHxW, out-shape: BxHxW
        lmap = self.pw_conv(local_map).squeeze(1)
        return lmap
        
class SimpleConvNet(nn.Module):
    def __init__(self, radius):
        super(SimpleConvNet, self).__init__()
        self.radius = radius
        self.layers = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size = 3),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 8, kernel_size = 3),
                nn.BatchNorm2d(8),
                nn.ELU(),
                nn.Conv2d(8, 1, kernel_size = (2*radius-3, radius-4)), #FCN-type idea.
            )
    
    def forward(self, amap):
        x = self.layers(amap).squeeze(-1)
        return x

convnet_nn = LocalConvNet(SimpleConvNet(7), 7).to(device)
convnet_fm =gnet_fm
convnet = ModelPair(convnet_nn, convnet_fm, 'convnet', True)
convnet_gen = lambda : ModelPair(LocalConvNet(SimpleConvNet(7), 7).to(device), convnet_fm, 'convnet', True)