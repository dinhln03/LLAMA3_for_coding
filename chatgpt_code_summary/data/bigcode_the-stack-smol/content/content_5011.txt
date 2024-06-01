from main import main
from src.common import fill_with

class Arguments:
   pass

# latex info
result_structure = '|c|c|c|c|c|c|'
test_structure   = '|c|c|c|c|c|c|c|'
table_body       = '''
\\begin{center}
\t\\begin{tabular}{%s}
\t\t\\hline \\rowcolor{brown!50}
%s
\t\\end{tabular}
\\end{center}\n
'''

# Util data for build the result tables
bots             = ['reactive', 'proactive']
test_data_order  = ['DIRTY-MEAN', 'CLEAN', 'TLE', 'FIRED']
result_headers   = '\t\tTest Id & Tipo de robot & Suciedad media & Ambiente limpio & Tiempo terminado & Despedido \\\\ '

# Util data for build the test table
prop           = ['rows', 'columns', 'babies', 'time', 'toys', 'dirty']
test_headers   = "\t\tTest Id & Filas & Columnas & Beb\\'es & Tiempo(t) & Obstaculos & Suciedad \\\\ \hline"

test = [
    # n,  m,  b,  t,  o,  d
    [10, 10,  6,  2, 20, 40],
    [10, 10,  6,  3, 20, 40],
    [10, 10,  6,  4, 20, 40],
    [10, 10,  6,  5, 20, 40],
    [10, 10,  6, 10, 20, 40],
    [10, 10,  6, 15, 20, 40],
    [ 7,  8,  7,  5, 10, 10],
    [ 7,  8,  7,  5, 14, 30],
    [20, 20, 16, 15,  0, 45],
    [20, 20, 16, 15, 30,  0],
    [20, 20,  5, 15,  0,  0],
    [ 7,  8,  6,  2, 20, 40],
]

def build_args():
    args             = Arguments()
    args.env         = 'house'
    args.level       = 'notset'
    args.log_file    = ''
    args.cicles      = 100
    args.repetitions = 100
    args.verbose     = False
    return args

def build_test_row(data, id):
    test_name = '\\multirow{%d}{*}{t%d}' % (len(data), id)
    empty     = ' ' * len(test_name)
    row       = ' & %s' * 5
    callback  = lambda: empty
    test_gen  = fill_with([test_name], callback) 

    table = []
    for (name, d), head in zip(data, test_gen):
        current_row = row % (name, *[str(d[key]) for key in test_data_order])
        table.append(f'\t\t{head}{current_row} \\\\ ')
    return '\\cline{2-6}\n'.join(table) + '\\hline '

def build_test_table(data):
    table = [test_headers]
    for i, t in enumerate(data):
        row = ' & '.join([str(v) for v in [f't{i}', *t]])
        table.append(f'\t\t{row} \\\\ \\hline')
    return '\n'.join(table)

def build_result_table(data):
    args = build_args()
    table = [result_headers]
    for i, t in enumerate(data):
        for attr in zip(prop, t):
            setattr(args, *attr)
        data = []
        for bot in bots:
            args.robot = bot
            data.append((bot, main(args)))
        table.append(build_test_row(data, i))
    return '\\hline\n'.join(table)

if __name__ == "__main__":
    print(table_body % (test_structure, build_test_table(test)))
    print(table_body % (result_structure, build_result_table(test)))
