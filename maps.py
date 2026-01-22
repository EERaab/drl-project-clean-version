from settings import *
from utils import *

def flood(width, height, seeds):
    map_matrix = np.zeros((width, height),dtype = "int8")
    enum_set = np.ndenumerate(map_matrix)
    L = len(seeds)
    for coordinate_pair in enum_set:
        index = min(range(L), key = lambda i: taxi_distance(coordinate_pair[0], seeds[i])) #can be optimized - no need to check all points.
        if (map_matrix[coordinate_pair[0]] == 0):
            map_matrix[coordinate_pair[0]] = index + 1

    return map_matrix

def idf(point, i):
    return i

#An alternative method: Should be preferrable when the set of seeds is more dense.
#expands outward from each seed in increasingly bigger radii
#stops updating for a seed when no new points were 'colored in'
#can be made a bit more efficient (maybe) by iterating over displacement vectors.
def flood_2(width, height, seeds, terrain_function = idf):
    matrix = np.zeros((width, height), dtype = "int8") - 1
    L = len(seeds)
    incomplete_list = [True for i in range(L)] #checks if the i:th element of the tessellation has been completed
    radius = 0
    while any(incomplete_list):
        for i in range(L):
            #for displacement in displacement_vectors
            # pt = seeds[i] + displacement
            #if inbounds(pt) && matrix[pt] == 0
            # do the thing
            if incomplete_list[i]:
                base_point = [seeds[i][0] + radius, seeds[i][1]]
                no_flips = True #we track if any tiles have been altered within this radius
                for point in circle_iterator(base_point, radius, width, height):
                    if ((matrix[point]) == -1): #-1 corresponds to an unassigned tile.
                        matrix[point] = terrain_function(point, i) #i+1# f(i, param, ...)
                        no_flips = False
                if no_flips:
                    incomplete_list[i] = False
        radius += 1
    return matrix

def disc_fill_map(width, height, dot_density, max_spread, value = 1):
    # dot_number ~ dot_density*volume
    partial_map = np.zeros((width, height), dtype = "int8")
    for j in range(int(dot_density*height*width)): #for the love of all that is holy set a small density - python doesnt do for loops.
        pt = [random.randrange(0, width), random.randrange(0, height)]
        radial_spread = random.randrange(1, max_spread+1)
        for pt in disc_iterator(pt, radial_spread, width, height):
            partial_map[pt] = value
    return partial_map

def terrain_from_underlying(underlying_map, point, i):
    return underlying_map[point[0], point[1], i]

def generate_simple_world(width, height, seed_nr, initial_position=False, max_dot_density = 0.05, min_dot_density = 0.0005, max_spread = 2, clear_radius = 3, verbose = False):
    seeds = []
    map_in_voronoi_channels = np.ones((width, height, seed_nr), dtype = "int8")
    if not initial_position:
        initial_position = (int((width-1)/2), int((height-1)/2))
    if verbose:
        print(f"Generating world with {seed_nr} seeds...")

    for j in range(seed_nr):
        #Central region should never be dense, so we ensure the central point is contained in the first region and set its tiles to be generally walkable
        if j == 0:
            seeds.append(initial_position)
            dot_density = min_dot_density
            map_in_voronoi_channels[:,:,j] = disc_fill_map(width, height, dot_density, 1)
            if verbose:
                print(f"{j}:th seed at {seeds[-1]} with dot density {dot_density}")
        else:
            new_seed = (random.randrange(width), random.randrange(height))
            while new_seed in seeds:
                new_seed = (random.randrange(width), random.randrange(height))
            seeds.append(new_seed)
            dot_density = random.uniform(min_dot_density, max_dot_density)
            map_in_voronoi_channels[:,:,j] = disc_fill_map(width, height, dot_density, max_spread)
            if verbose:
                print(f"{j}:th seed at {seeds[-1]} with dot density {dot_density}")


    world_map = flood_2(width, height, seeds, terrain_function = lambda pt, j : terrain_from_underlying(underlying_map = map_in_voronoi_channels, point = pt, i = j))
    world_map = clear_initial_tiles(world_map, initial_position, width, height, clear_radius)
    return world_map

def clear_initial_tiles(wmap, ini_tile, width, height, clear_radius):
    for pt in disc_iterator([ini_tile[0], ini_tile[1]], clear_radius, width, height):
        wmap[pt] = 0
    return wmap