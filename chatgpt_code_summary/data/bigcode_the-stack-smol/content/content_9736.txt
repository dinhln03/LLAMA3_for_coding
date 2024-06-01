from collections import defaultdict

from advent_2021.helpers import get_input


def dfs(
    caves: dict[str, list[str]],
    current: str,
    visited: set[str] | None = None,
) -> int:
    if current == "end":
        return 1

    nb_paths = 0
    if visited is None:
        visited = set()
    for cave in caves[current]:
        if cave in visited:
            continue
        nb_paths += dfs(
            caves, cave, visited | {current} if current.islower() else visited
        )

    return nb_paths


if __name__ == "__main__":
    caves: dict[str, list[str]] = defaultdict(list)
    for line in get_input():
        caves[line.split("-")[0]].append(line.split("-")[1])
        caves[line.split("-")[1]].append(line.split("-")[0])
    print(dfs(caves, "start"))
