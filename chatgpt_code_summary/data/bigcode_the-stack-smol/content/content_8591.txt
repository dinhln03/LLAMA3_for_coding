def main(x):
    matrix = []
    exit_path = []
    for i in range(0, x):
        j = list(input())
        if 'e' in j:
            y = j.index("e")
            exit_path.append(i)
            exit_path.append(y)
            j[y] = "-"
        matrix.append(j)

    row, col = 0, 0
    matrix[row][col] = "S"
    path = []
    searching_path(matrix, path, exit_path, row, col)


def searching_path(m, path, exit_path, i, j):
    r, c = len(m), len(m[0])
    exit_row, exit_col = exit_path

    # If destination is reached print
    if i == exit_row and j == exit_col:
        print("".join(e for e in path[1:]) + m[i][j])
        m[exit_row][exit_col] = "-"
        return

    # explore
    path.append(m[i][j])

    # move down
    if 0 <= i + 1 <= r - 1 and 0 <= j <= c - 1 and m[i + 1][j] == "-":
        m[i + 1][j] = "D"
        searching_path(m, path, exit_path, i + 1, j)

    # move right
    if 0 <= i <= r - 1 and 0 <= j + 1 <= c - 1 and m[i][j + 1] == '-':
        m[i][j + 1] = 'R'
        searching_path(m, path, exit_path, i, j + 1)

    # move left
    if 0 <= i <= r - 1 and 0 <= j - 1 <= c - 1 and m[i][j - 1] == '-':
        m[i][j - 1] = "L"
        searching_path(m, path, exit_path, i, j - 1)

    # move up
    if 0 <= i - 1 <= r - 1 and 0 <= j <= c - 1 and m[i - 1][j] == '-':
        m[i - 1][j] = "U"
        searching_path(m, path, exit_path, i - 1, j)

    # if none of the above is explorable or invalid index backtrack
    path.pop()


main(3)
