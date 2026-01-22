from settings import *
from utils import *
from mdp import *

class ModelPair:
    def __init__(self, nn_model, feature_mapping, name, is_mf):
        self.name = name
        self.nn_model = nn_model
        self.feature_mapping = feature_mapping
        self.is_multi_feature = is_mf
    
    def save(self, custom_name = None):
        if custom_name:
            torch.save(self.nn_model.state_dict(), custom_name + '.pth')
            return None
        torch.save(self.nn_model.state_dict(), self.name + '.pth')
        return None
    
    def load(self, custom_name = None):
        try:
            if custom_name:
                self.nn_model.load_state_dict(torch.load(custom_name + '.pth', weights_only=True))
                return None
            self.nn_model.load_state_dict(torch.load(self.name + '.pth', weights_only=True))
            return None
        except:
            print("No model parameters saved for this model.")

    def trainable_parameters(self):
        return trainable_parameters(self.nn_model)

#This particular block is used in a few places.
class Tile3LayerConvolution(nn.Module):
    def __init__(self):
        super(Tile3LayerConvolution, self).__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size = 1, padding = 'valid'),
                nn.ELU(),
                nn.Conv2d(6, 1, kernel_size = 1, padding = 'valid'),
                nn.ELU(),
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        return x

def test_initialize_model(model):
    world = dense_world()
    state = initialize_state(world)
    model.nn_model.eval()
    try:
        model_is = model.feature_mapping.initialize_internal_state(state)
        print("Model internal state initialized...")
    except:
        print("Model failed to initialize internal state")
        model.nn_model.train()
        return None
    try:
        model_feat = model.feature_mapping.get_features(state, model_is)
        print("Model feature computed...")
        model_feat_type = type(model_feat)
        print(f"Feature type: {model_feat_type}")
    except:
        print("Model failed to compute feature internal state")
        model.nn_model.train()
        return None
    try:
        with torch.no_grad():
            if model_feat_type == tuple:
                out = model.nn_model(*model_feat)
            else:            
                out = model.nn_model(model_feat)
            print(f"Model outcome on initial state: {out}")
    except:
        print("Failed to run model over features, outputting features.")
        model.nn_model.train()
        return model_feat
    model.nn_model.train()
    model.trainable_parameters()
    print("Model passed")
    