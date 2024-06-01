import numpy as np
from shapely import geometry

def shrink(coords: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray]:
    """Shrinks a 2D polygon by a given distance.

    The coordinates of the polygon are expected as an N x 2-matrix,
    and a positive distance results in inward shrinking.
    
    An empty set is returned if the shrinking operation removes all
    original elements.

    Args:
        coords: A matrix of coordinates.
        dist: The distance to shrink by.

    Returns:
        A tuple containing the x, y coordinates of the original set, as
        well as the x and y coordinates of the shrunken set, in that
        order.
    """
    my_polygon = geometry.Polygon(coords)
    xy = my_polygon.exterior.xy
    
    my_polygon_shrunken = my_polygon.buffer(-dist)
    
    try:
        xys = my_polygon_shrunken.exterior.xy
    except AttributeError:
        xys = ([0], [0]) # Empty set
    
    return (*xy, *xys)

def hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    """Computes the Hausdorff distance between two 2D polygons.

    Args:
        A: A matrix defining the first polygon.
        B: A matrix defining the second polygon.
    
    Returns:
        A float representing the Hausdorff distance.
    """
    return geometry.Polygon(A).hausdorff_distance(geometry.Polygon(B))

def read_polygon(file: str) -> np.ndarray:
    """Reads a polygon from a table.

    Args:
        file: Path to a file containing a plain text, tab-separated
            table with scalars.
    
    Returns:
        A matrix containing the data in the file.
    """
    return np.genfromtxt(file)

if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Distance to shrink by
    dh = 0.01

    x, y, xs, ys = shrink(read_polygon('example.txt'), dh)

    ax = plt.subplot()
    ax.grid(which='major', alpha=0.5, color='k')
    ax.grid(which='minor', alpha=0.3, color='k', linestyle=':')
    ax.minorticks_on()
    ax.set_axisbelow(True)

    ax.fill(x, y, color='b', facecolor='lightskyblue',
        edgecolor='dodgerblue', label='Original', alpha=0.75)
    ax.fill(xs, ys, facecolor='mediumseagreen', edgecolor='forestgreen',
        label='Shrunk', alpha=0.75)
    ax.set_aspect('equal')
    ax.legend()

    golden = 0.01017601435813135

    assert(np.isclose(
        hausdorff(np.vstack([x, y]).T, np.vstack([xs, ys]).T),
        golden
    ))

    print("SUCCESS")
    print(f'Area original: {geometry.Polygon(np.vstack([x, y]).T).area:.6f}')
    print(f'Area shrunk: {geometry.Polygon(np.vstack([xs, ys]).T).area:.6f}')
    plt.show()