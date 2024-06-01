from ourstylePy import data

def our_colours(colours=[]):
    '''
    Extract hexcodes for our colours
    If passed a sting, returns the matching hexcode.
    If passed a list, returns a list of hexcodes.
    Method from https://drsimonj.svbtle.com/creating-corporate-colour-palettes-for-ggplot2.
    - colours, list of strings

    Examples:
    data.our_colours_raw
    our_colours()
    our_colours('green', 'blue', 'green')
    our_colours('not a colour', 'also not a colour', 'green')
    our_colors('blue')
    '''
    if len(colours) == 0:
        return data.our_colours_raw
    elif isinstance(colours, str):
        return data.our_colours_raw[colours]
    else:
        return [data.our_colours_raw[i] for i in colours]

def our_colors(colours=[]):
    '''
    Alias for our_colours()
    '''
    return our_colours(colours)
