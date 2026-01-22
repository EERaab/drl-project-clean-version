from settings import *
from utils import *
from maps import *

class GridWorldEnvironment:  #Effectively p(s', r|s, a).

    def __init__(self, world_map, boundary_padding_width, initial_position = False, exploration_radius = 2, bad_walk_penalty = -1, new_tile_reward = 1, reward_bias = -2):
        self.world_map = np.pad(world_map, ((boundary_padding_width, boundary_padding_width), (boundary_padding_width, boundary_padding_width)), constant_values = 1)
        self.exploration_radius = exploration_radius
        self.initial_position = initial_position
        self.boundary_padding_width = boundary_padding_width

        sh = self.world_map.shape
        self.width = sh[0]
        self.height = sh[1]
        self.reward_bias = reward_bias
        if not initial_position:
            self.initial_position = (int((sh[0]-1)/2), int((sh[1]-1)/2))
        else:
            self.initial_position = initial_position
        self.bad_walk_penalty = bad_walk_penalty + reward_bias
        self.new_tile_reward = new_tile_reward

    def reward(self, new_tiles_seen): #this could be far more interesting - this is the most basic form of reward.
        return len(new_tiles_seen)*self.new_tile_reward + self.reward_bias

    def walkable(self, tile): #if we do multi-channels for the world_map we need to change this
        if self.world_map[tile] == 1:
            return False
        return True

    def plot(self):
        plot_world(self)

def initialize_state(environment):
    agent_map = np.zeros((2, environment.width, environment.height), dtype = "int8")

    #Set initial position
    position = environment.initial_position

    #Set agents initial_exploration
    for point in disc_iterator([environment.initial_position[0], environment.initial_position[1]], environment.exploration_radius, environment.width, environment.height):
        agent_map[0, point[0], point[1]] = 1 #the agent has 'seen' the tile
        agent_map[1, point[0], point[1]] = environment.world_map[point] #and the data the agent has 'seen' is the same as that of the world.

    boundary_padding_width = environment.boundary_padding_width
    #the agent also should pad its map around the edges. It treats the tiles as 'seen' and 'unwalkable'.
    agent_map[0,:boundary_padding_width, :] = 1
    agent_map[1,:boundary_padding_width, :] = environment.world_map[:boundary_padding_width, :]
    agent_map[0,-boundary_padding_width:, :] = 1
    agent_map[1,-boundary_padding_width:, :] = environment.world_map[-boundary_padding_width:, :]
    agent_map[0,:, :boundary_padding_width] = 1
    agent_map[1,:, :boundary_padding_width] = environment.world_map[:, :boundary_padding_width]
    agent_map[0,:, -boundary_padding_width:] = 1
    agent_map[1,:, -boundary_padding_width:] = environment.world_map[:, -boundary_padding_width:]

    return State(position, agent_map)

class State:
    def __init__(self, position, agent_map):
        self.position = position
        self.agent_map = agent_map

    def initialize_state(self, environment):
        agent_map = np.zeros((2, environment.width, environment.height), dtype = "int8")

        #Set initial position
        self.position = environment.initial_position

        #Set agents initial_exploration
        for point in disc_iterator([environment.initial_position[0], environment.initial_position[1]], environment.exploration_radius, environment.width, environment.height):
            agent_map[0, point[0], point[1]] = 1 #the agent has 'seen' the tile
            agent_map[1, point[0], point[1]] = environment.world_map[point] #and the data the agent has 'seen' is the same as that of the world.

        boundary_padding_width = environment.boundary_padding_width
        #the agent also should pad its map around the edges. It treats the tiles as 'seen' and 'unwalkable'.
        agent_map[0,:boundary_padding_width, :] = 1
        agent_map[1,:boundary_padding_width, :] = environment.world_map[:boundary_padding_width, :]
        agent_map[0,-boundary_padding_width:, :] = 1
        agent_map[1,-boundary_padding_width:, :] = environment.world_map[-boundary_padding_width:, :]
        agent_map[0,:, :boundary_padding_width] = 1
        agent_map[1,:, :boundary_padding_width] = environment.world_map[:, :boundary_padding_width]
        agent_map[0,:, -boundary_padding_width:] = 1
        agent_map[1,:, -boundary_padding_width:] = environment.world_map[:, -boundary_padding_width:]
        self.agent_map = agent_map

    def transition(self, action, environment):
        if action == 0: #increment the first index
            new_point = (self.position[0] + 1, self.position[1])
        elif action == 1:
            new_point = (self.position[0], self.position[1] + 1)
        elif action == 2:
            new_point = (self.position[0] - 1, self.position[1])
        elif action == 3:
            new_point = (self.position[0], self.position[1] - 1)

        if in_bounds(new_point, environment.width, environment.height) and environment.walkable(new_point):
            self.position = new_point
        else:
            return environment.bad_walk_penalty

        new_tiles_seen = []
        #We see new tiles in a taxi-cab half circle.
        for point in directional_half_circle_iterator(self.position, action, environment.exploration_radius):
            if in_bounds(point, environment.width, environment.height) and self.agent_map[0, point[0], point[1]] == 0:
                self.agent_map[0, point[0], point[1]] = 1
                self.agent_map[1, point[0], point[1]] = environment.world_map[point]
                new_tiles_seen.append(point)

        return environment.reward(new_tiles_seen)

    def flatten_map(self):
        flat_map = np.multiply(self.agent_map[0,:,:], (self.agent_map[1,:,:] + 1) ) - 1
        return flat_map

    def plot(self):
        plot_state(self)

    def copy(self):
        namap = self.agent_map.copy()
        return State(self.position, namap)


#Default world
def_world = lambda : GridWorldEnvironment(generate_simple_world(WORLD_BASE_SHAPE[0], WORLD_BASE_SHAPE[1], WORLD_SEED_NR), WORLD_BOUNDARY_PADDING)
em_world = lambda : GridWorldEnvironment(generate_simple_world(WORLD_BASE_SHAPE[0], WORLD_BASE_SHAPE[1], 1, max_dot_density = 0.0, min_dot_density = 0.0), WORLD_BOUNDARY_PADDING)
dense_world = lambda : GridWorldEnvironment(generate_simple_world(49, 49, 10), WORLD_BOUNDARY_PADDING)

#Smallworld, for testing
sm_world = lambda : GridWorldEnvironment(generate_simple_world(11, 11, 3, (5, 5)), 2)
