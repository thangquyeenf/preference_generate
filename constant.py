

priority_list = [(3, 3), (2, 3), (1, 3), (0, 3), 
                 (0, 2), (1, 2), (2, 2), (3, 2),
                 (3, 1), (2, 1), (3, 0), (2, 0),
                 (1, 0), (1, 1), (0, 1), (0, 0)]

def get_priority(coord):
    return priority_list.index(coord)

def compare_coordinates(coord1, coord2) -> bool:
    priority1 = get_priority(coord1)
    priority2 = get_priority(coord2)
    return priority1 < priority2