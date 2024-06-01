from collections import defaultdict


def list_to_map(Xs, ys):
    labels_map = defaultdict(list)
    for x, y in list(zip(Xs, ys)):
        labels_map[y].append(x)
    return labels_map