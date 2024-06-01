from ..utils import sortkey, capitalize_first

FIGURE_TEX_TEMPLATE = r'\hwgraphic{{{path}}}{{{headword}}}{{{attribution}}}'
# change to {filename} if you want to specify full paths.
FIGURE_PATH_TEMPLATE = r'figures/ill-{filename}'


class Image(object):
    type = 'img'

    def sk(self):
        return sortkey(self.hw)

    def __init__(self, hw='', img_src='', img_attrib=''):
        super().__init__()
        self.hw = hw
        self.img_src = img_src
        self.img_attrib = img_attrib

    def __repr__(self):
        return "(Image of '{headword}')".format(
            headword=self.hw
        )

    def render(self, settings={}):
        figure_path = FIGURE_PATH_TEMPLATE.format(filename=self.img_src)
        return FIGURE_TEX_TEMPLATE.format(
            headword=capitalize_first(self.hw),
            path=figure_path,
            attribution=self.img_attrib
        )
