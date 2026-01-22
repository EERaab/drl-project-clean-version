from settings import *

def plot_world(world):
    colors_list = ['#33cc33','#0099ff']
    cmap = colors.ListedColormap(colors_list)
    try:
        plt.imshow(world.world_map, cmap=cmap, origin="lower")
    except:
        plt.imshow(world, cmap=cmap, origin="lower")
    plt.show()

def plot_state(state, colors_list = ['#000000', '#33cc33','#0099ff']):
    try:
        smap = state.flatten_map()
    except:
        smap = state.agent_map
    cmap = colors.ListedColormap(colors_list)
    plt.imshow(smap, cmap=cmap, origin="lower")
    plt.plot(state.position[1], state.position[0], '-ro', color = '#fff200')
    plt.show()


def plot_local(state, radius = WORLD_BOUNDARY_PADDING):
    fsmap = state.flatten_map()
    window = fsmap[state.position[0]-radius:state.position[0]+radius+1, state.position[1]-radius:state.position[1]+radius+1]
    if np.max(window) == np.min(window) + 1:
        colors_list = ['#000000', '#33cc33']
    else:
        colors_list = ['#000000', '#33cc33', '#0099ff']
    cmap = colors.ListedColormap(colors_list)
    plt.imshow(window, cmap=cmap, origin="lower")
    plt.plot([radius], [radius], '-o', color = '#fff200')
    plt.show()


def plot_episode(states, assesments, radius = WORLD_BOUNDARY_PADDING, pause = 0.5):
    #This is goofy, should use the animation package.
    plt.figure()
    i=0
    for state in states:
        plt.clf()
        fsmap = state.flatten_map()
        window = fsmap[state.position[0]-radius:state.position[0]+radius+1, state.position[1]-radius:state.position[1]+radius+1]
        if np.max(window) == np.min(window) + 1:
            colors_list = ['#000000', '#33cc33']
        else:
            colors_list = ['#000000', '#33cc33', '#0099ff']
        cmap = colors.ListedColormap(colors_list)
        plt.cla()
        plt.imshow(window, cmap=cmap, origin="lower")
        plt.plot([radius], [radius], '-h', color = '#fff200')
        if i:
            dx= state.position[0]-prev_state.position[0]
            dy= state.position[1]-prev_state.position[1]
            plt.arrow(radius-dy, radius-dx, dy, dx, head_starts_at_zero = True, width = 0.005)
        plt.title(f"Assesment: {assesments[i]}")
        plt.pause(pause)  # pause a bit so that plots are updated
        display.display(plt.gcf())
        display.clear_output(wait=True)
        prev_state = state
        i+=1