from settings import *
from features import *
from utils import *
from models import *
from gnet_features import *
from gnet_model import *
from symmetrization import *


class BaseGlobalPathfinder(nn.Module):
    def __init__(self, radius, dropout = 0.1):
        super(BaseGlobalPathfinder, self).__init__()
                
        self.layers = nn.Sequential(
                nn.Flatten(start_dim = -2),
                nn.Linear((2*radius+1)**2, 6*(2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear( 6*(2*radius+1), 2*(2*radius+1)),
                nn.Dropout(dropout),
                nn.ELU(),
                nn.Linear(2*(2*radius+1), 4, bias=False)
            )
    
    def forward(self, amap):
        x = self.layers(amap)
        return x
