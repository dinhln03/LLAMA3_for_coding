def pixel(num):
    def f(s):
        return s + '\033[{}m  \033[0m'.format(num)
    return f

def new_line(s):
    return s + u"\n"

def build(*steps, string=""):
    for step in steps:
        string = step(string)
    return string

def main():
    cyan = pixel(46)
    space = pixel('08')
    heart = [new_line,
            space, space, cyan, cyan, space, space, space, cyan, cyan, new_line,
            space, cyan, cyan, cyan, cyan, space, cyan, cyan, cyan, cyan, new_line,
            cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, new_line,
            cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, new_line,
            cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, new_line,
            space, cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, cyan, new_line,
            space, space, cyan, cyan, cyan, cyan, cyan, cyan, cyan, new_line,
            space, space, space, cyan, cyan, cyan, cyan, cyan, new_line,
            space, space, space, space, cyan, cyan, cyan, new_line,
            space, space, space, space, space, cyan, new_line]

    print(build(*heart))

if __name__ == '__main__':
    main()