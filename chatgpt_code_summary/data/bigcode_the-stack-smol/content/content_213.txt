'''
Equations are given in the format A / B = k, where A and B are variables represented as strings, and k is a real number (floating point number). Given some queries, return the answers. If the answer does not exist, return -1.0.

Example:
Given a / b = 2.0, b / c = 3.0.
queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? .
return [6.0, 0.5, -1.0, 1.0, -1.0 ].

The input is: vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries , where equations.size() == values.size(), and the values are positive. This represents the equations. Return vector<double>.

According to the example above:

equations = [ ["a", "b"], ["b", "c"] ],
values = [2.0, 3.0],
queries = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ].
The input is always valid. You may assume that evaluating the queries will result in no division by zero and there is no contradiction.

'''

class Solution(object):
    def buildGraph(self, edges, vals):
        """
        :type edge: List[[str, str]]
        :type vals: List[Double]
        :rtype: dict[dict]
        """
        import collections
        graph = collections.defaultdict(dict)
        for index, val in enumerate(vals):
            start = edges[index][0]
            end = edges[index][1]
            graph[start][end] = val
            graph[end][start] = 1 / val
        return graph

    def insert(self, start, end, val):
        self.graph[start][end] = val
        self.graph[end][start] = 1 / val

    def search(self, start, end):
        val = 1.0
        visited = dict()
        size = len(self.graph)
        mark = set()
        mark.add(start)
        visited[start] = 1.0
        while (len(mark) > 0) and (end not in visited):
            src = mark.pop()
            for (dest, val) in self.graph[src].items():
                if dest not in visited:
                    mark.add(dest)
                    visited[dest] = visited[src] * val
        return visited.get(end, -1.0)

    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        self.graph = self.buildGraph(equations, values)
        output = list()
        for (start, end) in queries:
            if start not in self.graph or end not in self.graph:
                output.append(-1.0)
                continue
            val = self.search(start, end)
            if val > 0:
                output.append(val)
                self.insert(start, end, val)
            else:
                output.append(-1.0)
        return output


solution = Solution()
equations = [["x1","x2"],["x2","x3"],["x3","x4"],["x4","x5"]]
values = [3.0,4.0,5.0,6.0]
queries = [["x1","x5"],["x5","x2"],["x2","x4"],["x2","x2"],["x2","x9"],["x9","x9"]]

print solution.calcEquation(equations, values, queries)
