"""

link: https://leetcode-cn.com/problems/smallest-rectangle-enclosing-black-pixels

problem: 给定 0, 1 矩阵，以及一个矩阵中为 1 的点坐标，求包含矩阵中所有的1的最小矩形面积

solution: 暴搜。忽略坐标，直接遍历所有节点，找到上下左右四个边界点，时间O(nm)。

solution-fix: 二分。将x轴投影到y轴，y轴投影到x轴，形成两个一维数组。显然数组形如下图。而 x, y 坐标为界，两侧各为非严格递增和递减
              1:      +------+
              0: -----+      +-----
              四次二分找到递增递减边界，时间复杂度 O(nlogn*mlogm)

"""

class Solution:
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        if not image or not image[0]:
            return 0
        n, m = len(image), len(image[0])
        a, b, c, d = n, m, 0, 0
        for i in range(n):
            for j in range(m):
                if image[i][j] == '1':
                    a = min(a, i)
                    b = min(b, j)
                    c = max(c, i)
                    d = max(d, j)
        return (c + 1 - a) * (d + 1 - b)

# ---
class Solution:
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        if not image or not image[0]:
            return 0
        n, m = len(image), len(image[0])

        def search_column(l: int, r: int, up: bool) -> int:
            k = r if up else l
            while l <= r:
                mid, mid_k = (l + r) >> 1, 0
                for i in range(m):
                    if image[mid][i] == '1':
                        mid_k = 1
                        break
                if mid_k:
                    k = min(k, mid) if up else max(k, mid)
                if mid_k ^ up:
                    l = mid + 1
                else:
                    r = mid - 1
            return k

        def search_row(l: int, r: int, up: bool) -> int:
            k = r if up else l
            while l <= r:
                mid, mid_k = (l + r) >> 1, 0
                for i in range(n):
                    if image[i][mid] == '1':
                        mid_k = 1
                        break
                if mid_k:
                    k = min(k, mid) if up else max(k, mid)
                if mid_k ^ up:
                    l = mid + 1
                else:
                    r = mid - 1
            return k

        a = search_column(0, x, True)
        b = search_row(0, y, True)
        c = search_column(x, n - 1, False)
        d = search_row(y, m - 1, False)
        return (c + 1 - a) * (d + 1 - b)
