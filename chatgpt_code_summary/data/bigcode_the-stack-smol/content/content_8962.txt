import numpy as np
import sys
path = sys.argv[1]


points = []
curves = []
polygons = []

with open(path, 'r') as f:
    line = f.readline()
    while line:
        line = line.strip()
        if len(line) <= 0:
            line = f.readline()
            continue
        tokens = line.split(' ')
        if tokens[0] == 'p':
            points.append([
                float(tokens[1]),
                float(tokens[2])])
        elif tokens[0] == 'c':
            tmp = []
            for i in range(1, len(tokens)):
                tmp.append(int(tokens[i]))
            curves.append(tmp)
        elif tokens[0] == 'poly':
            tmp = []
            for i in range(1, len(tokens)):
                tmp.append(int(tokens[i]))
            polygons.append(tmp)
        else:
            assert(False)

        line = f.readline()

points = np.array(points)
mmin = np.min(points, axis=0)
mmax = np.max(points, axis=0)

with open("test.eps", 'w') as f:
    f.write("%!PS-Adobe-3.0 EPSF-3.0\n")
    f.write("%%BoundingBox: {} {} {} {}\n\n".format(mmin[0], mmin[1], mmax[0], mmax[1]))
    f.write("%%Pages: 1\n")
    f.write("%%Page: 1 1\n")
    f.write("/show-ctr {\ndup stringwidth pop\n -2 div 0\n rmoveto show\n} def\n\n 2 setlinejoin\n\n")

    f.write("255 0 0 setrgbcolor\n")
    f.write("1 setlinewidth\n\n")

    for poly in polygons:
        first = True
        for curve_idx in poly:
            curve = curves[curve_idx]
            if first:
                f.write("{} {} moveto\n".format(points[curve[0], 0], points[curve[0], 1]))
                first = False

            if len(curve) == 4:
                f.write("{} {} {} {} {} {} curveto\n".format(
                    points[curve[1], 0], points[curve[1], 1],
                    points[curve[2], 0], points[curve[2], 1],
                    points[curve[3], 0], points[curve[3], 1]))
            elif len(curve) == 2:
                f.write("{} {} lineto\n".format(
                    points[curve[1], 0], points[curve[1], 1]))
            else:
                print(curve)
                assert(False)

        f.write("stroke\n\n\n")


