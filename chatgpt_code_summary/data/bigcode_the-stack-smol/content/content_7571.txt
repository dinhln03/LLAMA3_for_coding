def dfs(graph, start, end):
    stack = [start]
    visited = []
    while stack:
        u = stack.pop()  # stack에서 아이템을 빼낸다.
        visited.append(u)
        if end in visited:
            return 1
        for v in graph[u]:
            if v not in visited and v not in stack:
                stack.append(v)
    return 0


t = int(input())
for i in range(t):
    graph = {}
    node, seg = map(int, input().split())
    for _ in range(seg):
        a, b = map(int, input().split())
        graph[a] = graph.get(a, []) + [b]
        if b not in graph:
            graph[b] = []
        # graph[b] = graph.get(b, []) + [a]
    start, end = map(int, input().split())
    print(f"#{i + 1} {dfs(graph, start, end)}")
