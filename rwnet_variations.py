from settings import *
from features import *
from utils import *
from models import *
from gnet_features import *
from gnet_model import *
from symmetrization import *

class SimpleLocalPathfinderNetWithDropout(nn.Module):
    def __init__(self, radius, dropout = 0.2):
        super(SimpleLocalPathfinderNetWithDropout, self).__init__()
                
        self.layers = nn.Sequential(
                nn.Linear(radius*(2*radius+1), 3*(2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear( 3*(2*radius+1), (2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear((2*radius+1), 1)
            )
    
    def forward(self, amap):
        x = self.layers(amap).squeeze(-1)
        return x

# The defining feature of rwnet is the 'SimpleLocalPathfinder' nn.Module. We adjust it in a few ways
class RW_deeper_1(nn.Module):
    def __init__(self, radius, dropout=0.2):
        super(RW_deeper_1, self).__init__()
                
        self.layers = nn.Sequential(
                nn.Linear(radius*(2*radius+1), 3*(2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear( 3*(2*radius+1), 3*(2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear( 3*(2*radius+1), (2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear((2*radius+1), 1)
            )
    
    def forward(self, amap):
        x = self.layers(amap).squeeze(-1)
        return x

#Two extra
class RW_deeper_2(nn.Module):
    def __init__(self, radius, dropout = 0.2):
        super(RW_deeper_2, self).__init__()
                
        self.layers = nn.Sequential(
                nn.Linear(radius*(2*radius+1), 3*(2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear( 3*(2*radius+1), 3*(2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear( 3*(2*radius+1), 3*(2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear( 3*(2*radius+1), (2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear((2*radius+1), 1)
            )
    
    def forward(self, amap):
        x = self.layers(amap).squeeze(-1)
        return x

#Wider hidden layers
class RW_wider(nn.Module):
    def __init__(self, radius, dropout = 0.2):
        super(RW_wider, self).__init__()
                
        self.layers = nn.Sequential(
                nn.Linear(radius*(2*radius+1), 6*(2*radius+1)),
                nn.Dropout(dropout),
                nn.Linear(6*(2*radius+1), 3*(2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear(3*(2*radius+1), 1)
            )
    
    def forward(self, amap):
        x = self.layers(amap).squeeze(-1)
        return x


#Wider and deeper
class RW_wider_deeper(nn.Module):
    def __init__(self, radius, dropout = 0.2):
        super(RW_wider_deeper, self).__init__()
                
        self.layers = nn.Sequential(
                nn.Linear(radius*(2*radius+1), 6*(2*radius+1)),
                nn.Dropout(dropout),
                nn.Linear(6*(2*radius+1), 6*(2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear(6*(2*radius+1), 3*(2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear(3*(2*radius+1), 1)
            )
    
    def forward(self, amap):
        x = self.layers(amap).squeeze(-1)
        return x

