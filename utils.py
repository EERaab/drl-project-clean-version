
def taxi_distance(point1, point2):
    return abs(point1[0]-point2[0]) + abs(point1[1]-point2[1])

def circle_iterator(final_point, radius, width, height):
    if radius == 0:
        yield (final_point[0], final_point[1])
    else:
        new_point = final_point
        for r in range(radius):
            new_point[0] -= 1
            new_point[1] += 1
            if in_bounds(new_point, width, height):
                yield (new_point[0], new_point[1])
        for r in range(radius):
            new_point[0] -= 1
            new_point[1] -= 1
            if in_bounds(new_point, width, height):
                yield (new_point[0], new_point[1])
        for r in range(radius):
            new_point[0] += 1
            new_point[1] -= 1
            if in_bounds(new_point, width, height):
                yield (new_point[0], new_point[1])
        for r in range(radius):
            new_point[0] += 1
            new_point[1] += 1
            if in_bounds(new_point, width, height):
                yield (new_point[0], new_point[1])

def in_bounds(point, width, height):
    if 0 <= point[0] < width:
        if 0 <= point[1] < height:
            return True
    return False

def disc_iterator(central_point, max_radius, width, height):
    for radius in range(max_radius + 1):
        pos = [central_point[0] + radius, central_point[1]]
        for point in circle_iterator(pos, radius, width, height):
            yield point

def directional_half_circle_iterator(center_point, direction, radius):
    if radius == 0:
        yield (center_point[0], center_point[1])
    else:
        if direction == 0:
            for dx in range(0, radius + 1):
                x = center_point[0] + dx
                if dx < radius:
                    dy = radius - dx
                    y = center_point[1] + dy
                    yield (x, y)
                    y = center_point[1] - dy
                    yield (x, y)
                else:
                    yield (x, center_point[1])
        elif direction == 1:
            for dy in range(0, radius + 1):
                y = center_point[1] + dy
                if dy < radius:
                    dx = radius - dy
                    x = center_point[0] + dx
                    yield (x, y)
                    x = center_point[0] - dx
                    yield (x, y)
                else:
                    yield (center_point[0], y)
        elif direction == 2:
            for dx in range(0, radius + 1):
                x = center_point[0] - dx
                if dx < radius:
                    dy = radius - dx
                    y = center_point[1] + dy
                    yield (x, y)
                    y = center_point[1] - dy
                    yield (x, y)
                else:
                    yield (x, center_point[1])
        elif direction == 3:
            for dy in range(0, radius + 1):
                y = center_point[1] - dy
                if dy < radius:
                    dx = radius - dy
                    x = center_point[0] + dx
                    yield (x, y)
                    x = center_point[0] - dx
                    yield (x, y)
                else:
                    yield (center_point[0], y)

def circle_iterator_inbounds(center_point, radius):
    if radius == 0:
        yield (center_point[0], center_point[1])
    else:
        new_point = [center_point[0] + radius, center_point[1]]
        for r in range(radius):
            new_point[0] -= 1
            new_point[1] += 1
            yield (new_point[0], new_point[1])
        for r in range(radius):
            new_point[0] -= 1
            new_point[1] -= 1
            yield (new_point[0], new_point[1])
        for r in range(radius):
            new_point[0] += 1
            new_point[1] -= 1
            yield (new_point[0], new_point[1])
        for r in range(radius):
            new_point[0] += 1
            new_point[1] += 1
            yield (new_point[0], new_point[1])

def window(amap, state, radius):
    return amap[state.position[0]-radius:state.position[0]+ 1+radius, state.position[1]-radius:state.position[1]+ 1+radius]