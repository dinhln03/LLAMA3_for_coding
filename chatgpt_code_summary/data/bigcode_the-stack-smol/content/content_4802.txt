# coding=utf-8

from src.tk import TK
import argparse

parser = argparse.ArgumentParser()
parser.register('type', 'bool', (lambda x: x.lower() in ('True', "yes", "true", "t", "1")))
parser.add_argument('--mode', default='main', help='')
args = parser.parse_args()

if args.mode == 'main':
    window = TK()
    window.start()
elif args.mode == 'N_0_2':
     from src.N_0_2 import KO
     ko = KO()
     ko.solve()
elif args.mode == 'test':
    from src.test import KO
    ko = KO()
    ko.solve()
elif args.mode == 'E_2_4':
    from src.E_2_4 import KO
    ko = KO()
    ko.solve()
elif args.mode == 'E_3_4':
    from src.E_3_4 import KO
    ko = KO()
    ko.solve()
elif args.mode == 'E_4_4':
    from src.E_4_4 import KO
    ko = KO()
    ko.solve()
elif args.mode == 'paper':
    from src.paper import KO
    ko = KO()
    ko.solve()
elif args.mode == 'N_5_6':
    from src.N_5_6 import KO
    ko = KO()
    ko.solve()
elif args.mode == 'E_10_4':
    from src.E_10_4 import KO
    ko = KO()
    ko.solve()
else:
    pass
