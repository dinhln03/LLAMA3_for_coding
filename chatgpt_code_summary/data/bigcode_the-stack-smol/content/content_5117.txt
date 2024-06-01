import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))

import files
import phylogeny


def main(argv):
    lines = files.read_lines(argv[0])
    taxa  = lines[0].split()
    table = lines[1:]

    print '\n'.join('{%s, %s} {%s, %s}' % (a1, a2, b1, b2) for ((a1, a2), (b1, b2)) in phylogeny.quartets(taxa, table))


if __name__ == "__main__":
    main(sys.argv[1:])
