from sys import argv, stdin


def cut(input_file, *args):
    options = process_options(*args)
    delimiter = d_option(options["-d"])
    lines = input_file.readlines()
    columns = [item.split(delimiter) for item in lines]
    scope = f_option(options["-f"], len(columns[0]))
    out_scope = []
    for x in scope:
        out_scope.append([column[x] for column in columns])

    pr = []
    for line in range(len(out_scope[0])):
        for rec in out_scope:
            pr.append(rec[line].strip())
        print(delimiter.join(pr), end='')
        pr.clear()
        print()


def process_options(options):
    out_opt = dict()
    last_key = ""
    for option in options:
        if option.startswith('-'):
            out_opt[option] = ""
            last_key = option
        else:
            out_opt[last_key] = option
    return out_opt


def f_option(params: str, file_size: int):
    if not params:
        return None
    inp = params.split('-') if '-' in params else params
    if '-' not in params and ',' not in params:
        return int(params)
    elif params.startswith('-'):
        return [x for x in range(0, int(inp[1]))]
    elif params.endswith('-'):
        return [x - 1 for x in range(int(inp[0]), file_size + 1)]
    elif ',' in params:
        return [int(x) for x in params.split(',')]
    else:
        return [x - 1 for x in range(int(inp[0]), int(inp[1]) + 1)]


def d_option(params):
    return params if params else ' '


cut(stdin, argv[1:])