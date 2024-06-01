import numpy as np
import matplotlib.pyplot as plt
import argparse

from camera import add_pmts

parser = argparse.ArgumentParser()
parser.add_argument('inputfile', help='A datafile created by the old SPECT camera')
parser.add_argument(
    '--outputfile', '-o', required=False,
    dest='outputfile',
    help='If given, save the image to outputfile'
)


if __name__ == '__main__':

    args = parser.parse_args()

    data = np.fromfile(args.inputfile, dtype='<u2')
    width = np.sqrt(data.size)
    assert width.is_integer()
    width = int(width)

    img = data.reshape((width, width))

    width = 60
    x_offset = 1
    y_offset = 0

    x0 = -width/2 - x_offset
    x1 = width/2 - x_offset
    y0 = -width/2 - y_offset
    y1 = width/2 - y_offset

    fig, ax = plt.subplots()

    ax.set_aspect(1)
    ax.set_axis_bgcolor('k')

    plot = ax.imshow(
        img,
        cmap='inferno',
        interpolation='nearest',
        extent=np.array([x0, x1, y0, y1]),
    )
    fig.colorbar(plot, label='Counts')

    add_pmts(ax=ax, linewidth=1.5)

    ax.set_xlim(-35, 35)
    ax.set_ylim(-26, 26)

    ax.set_xlabel('$x \,/\, \mathrm{cm}$')
    ax.set_ylabel('$y \,/\, \mathrm{cm}$')

    if args.outputfile:
        fig.savefig(args.outputfile, dpi=300)
    else:
        plt.show()
