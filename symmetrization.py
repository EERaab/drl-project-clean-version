from settings import *


# Let x denote a point in a feature space, and D one of four principal directions in this space
# Given a function f(x, D) that is invariant under rotations and  reflections of the space , there exists a function
# g such that for any (x,D) it holds that f(x, D) = f(Rx,0) = g(Rx) where R is a rotation matrix determined by D
# and it observes the symmetry g(y) = g(Ly) where L is a reflection over the 0-direction axis.
# Now if F(x) = [f(x,0), f(x,1), f(x,2), f(x,3)] then F is not invariant but equivariant, and we get
# F(Rx) = [f(Rx,0), f(Rx,1), f(Rx,2), f(Rx, 3)] = [g(Rx), g(R2x), g(R3x), g(R4x)] = [f(x,1),f(x,2),f(x,3), f(x,0)] = LinComb(F(x)). 
# This is the function we shall actually work with. 
# Conversely, given a function g which observes reflection symmetry g(y) = g(Ly) we can trivially construct the corresponding f. 
# In particular, if h(y) is *any* function then we that g(y) = h(y)+h(Ly) is trivially reflection invariant.
# Thus F(x) = [f(x,0), f(x,1), f(x,2), f(x,3)] = [g(x),g(Rx), g(R2x), g(R3x)] = [h(x) + h(Lx), h(Rx) + h(LRx), h(R2x) + h(LR2x), h(R3x) + h(LR3x)] observes all required symmetries.

#Takes any input map (BxHxW) and maps it to a (BxDxRxHxW)
def map_symmetrizer(any_map, rot_dims = (-2,-1), stack_dim = 1):
    d0 = any_map
    d1 = any_map.rot90(k=1,dims = rot_dims)
    d2 = any_map.rot90(k=2,dims = rot_dims)
    d3 = any_map.rot90(k=3,dims = rot_dims)
    dir_sym_map = torch.stack((d0,d1,d2,d3), dim = stack_dim)
    refl1 = dir_sym_map.flip(dims = (-1,))
    tot_sym = torch.stack((dir_sym_map, refl1), dim = stack_dim + 1 )
    return tot_sym

#Now any function that we apply condenses the last two dimensions to zero, and we symmetrize across reflections.
#In practice, h is now applied to condense away the HxW.
#Input map (BxDxR) output BxD
def map_combiner(sym_map):
    return torch.add(sym_map[:,:, 0],sym_map[:,:, 1])/2 #normalized here to avoid issues with activation functions

def window_directional_split_with_reflection(window_tensor, radius, stack_along_dim = 1):
    dir0 = window_tensor[:,radius+1:,:]
    dir1 = window_tensor[:,:,radius+1:].rot90(k=-1, dims=(1,2))
    dir2 = window_tensor[:,:radius,:].rot90(k=-2, dims=(1,2))
    dir3 = window_tensor[:,:,:radius].rot90(k=-3, dims=(1,2))
    #Stacking
    X = torch.stack((dir0,dir1,dir2,dir3), dim = stack_along_dim)
    #Reflection symmetrization 
    X = torch.stack((X, X.flip(dims = (-1,))), dim = stack_along_dim + 1)
    return X
