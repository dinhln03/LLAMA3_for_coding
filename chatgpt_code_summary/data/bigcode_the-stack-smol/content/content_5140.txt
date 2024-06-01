
# Space: O(n)
# Time: O(n!)


class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        self.data = characters
        self.res = self.combine(self.data, combinationLength)
        self.counter = 0
        self.res_count = len(self.res)

    def next(self) -> str:
        if self.hasNext():
            res = self.res[self.counter]
            self.counter += 1
            return res

    def hasNext(self) -> bool:
        return self.counter < self.res_count

    def combine(self, data, length):
        if length > len(data): return []

        def dfs(data, index, temp_res, res, length):
            if len(temp_res) == length:
                res.append(temp_res)
                return

            for i in range(index, len(data)):
                temp_res += data[i]
                dfs(data, i + 1, temp_res, res, length)
                temp_res = temp_res[:-1]
            return res

        return dfs(data, 0, '', [], length)




