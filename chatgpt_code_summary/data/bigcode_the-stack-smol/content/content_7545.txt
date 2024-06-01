from amaru.utilities import constants


def generate_subsets(current_tree_bottom):
    current_distances = []
    subsets = []
    current_point = 0
    while current_point < len(current_tree_bottom) - 1:
        current_distances.append(current_tree_bottom[current_point + 1][1] - current_tree_bottom[current_point][1])
        current_point = current_point + 1

    # remove similar splits causesd by floating point imprecision
    for i in range(len(current_distances)):
        current_distances[i] = round(current_distances[i], 10)

    split_points = list(set(current_distances))  # all possible x-distances between bottom blocks

    for i in split_points:  # subsets based on differences between x-distances
        current_subset = []
        start_point = 0
        end_point = 1
        for j in current_distances:
            if j >= i:
                current_subset.append(current_tree_bottom[start_point:end_point])
                start_point = end_point
            end_point = end_point + 1

        current_subset.append(current_tree_bottom[start_point:end_point])

        subsets.append(current_subset)

    subsets.append([current_tree_bottom])

    return subsets


# finds the center positions of the given subset

def find_subset_center(subset):
    if len(subset) % 2 == 1:
        return subset[(len(subset) - 1) // 2][1]
    else:
        return (subset[len(subset) // 2][1] - subset[(len(subset) // 2) - 1][1]) / 2.0 + subset[(len(subset) // 2) - 1][
            1]


# finds the edge positions of the given subset

def find_subset_edges(subset):
    edge1 = subset[0][1] - (constants.blocks[str(subset[0][0])][0]) / 2.0 + constants.edge_buffer
    edge2 = subset[-1][1] + (constants.blocks[str(subset[-1][0])][0]) / 2.0 - constants.edge_buffer
    return [edge1, edge2]
