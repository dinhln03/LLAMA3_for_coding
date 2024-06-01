"""

    *Element Height*

"""

from strism._geoshape import Pixel

from ._dimension import Dimension

__all__ = ["ElementHeight"]


class ElementHeight(
    Pixel,
    Dimension,
):
    def __init__(
        self,
        height: int,
    ):
        super(ElementHeight, self).__init__(
            height,
        )
