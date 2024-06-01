"""
Provides `to_ltx` to convert numpy arrays to LaTeX.
"""

import numpy as np

from numpyarray_to_latex.utils import (
            math_form,
        )

def to_ltx(a,
           fmt='{:6.4f}',
           latexarraytype='array',
           imstring='i',
           is_row_vector=True,
           mathform=True,
           brackets='()',
           mark_elements=[],
           mark_color='pink',
           separate_columns=[],
           separate_rows=[],
           ):
    r"""
    Return a LaTeX array given a numpy array.

    Parameters
    ----------
    a : numpy.ndarray
    fmt : str, default = '{:6.2f}'
        python 3 formatter, optional-
        https://mkaz.tech/python-string-format.html
    latexarraytype : str, default = 'array'
        Any of

        .. code:: python

            "array"
            "pmatrix"
            "bmatrix"
            "vmatrix"
            "Vmatrix"
            "Bmatrix"

        if "array", you can specifiy the brackets
        with the keyword ``brackets``.
    imstring : str, default = 'i'
        Character to use to represent the imaginary unit.
        Usually ``'i'`` or ``'j'``
    is_row_vector : bool, default = True
        If the array is 1D, should the output be
        a row (True) or column (False) vector?
    mathform : bool, default = True
        wether to convert strings like ``1e+05``
        to ``1\times10^{5}``.
    brackets : iterable, default = '()'
        which brackets to use to wrap the matrix
        (must be two elements long).
        Use ``brackets = None`` if you don't want
        any brackets around the array.
    mark_elements : list, default = []
        list of tuples containing element indices that
        should be marked with a colorbox.
    mark_color : str, default = 'pink'
        The color with which to mark matrix elements.
    separate_columns : list, default = []
        list of column indices before which a vertical
        line should be drawn
    separate_rows : list, default = []
        list of row indices before which a horizontal
        line should be drawn

    Returns
    -------
    out: str
        Formatted LaTeX string

    Examples
    --------
    >>> from numpyarray_to_latex import to_ltx
    >>> tex = to_ltx([[2.,2.],[2.,2.]])
    >>> print(tex)
    \left(
    \begin{array}{}
        2.00 &    2.00\\
        2.00 &    2.00
    \end{array}
    \right)

    """
    a = np.array(a)

    if len(a.shape) > 2:
        raise NotImplementedError('Arrays having more than two dimensions cannot be converted.')

    if mark_elements is None:
        mark_elements = []

    if a.ndim == 2 and len(mark_elements)>0 and not all([hasattr(mark,'__len__') for mark in mark_elements]):
        raise ValueError("If the array is 2D, ``mark_elements`` should be 2D as well, but isn't")

    if len(a.shape) == 1:
        if len(mark_elements)>0 and hasattr(mark_elements[0],'__len__'):
            raise ValueError("If the array is 1D, ``mark_elements`` should be 1D as well, but isn't.")
        a = np.array([a])
        if is_row_vector is False:
            a = a.T
            mark_elements = [ (mark,0) for mark in mark_elements]
        else:
            mark_elements = [ (0,mark) for mark in mark_elements]

    if isinstance(mark_elements, np.ndarray):
        mark_elements = mark_elements.tolist()
        mark_elements = [ tuple(row) for row in mark_elements ]


    nrow, ncol = a.shape

    out = ''

    if brackets is not None and latexarraytype not in [
                "bmatrix",
                "pmatrix",
                "vmatrix",
                "Bmatrix",
                "Vmatrix",
            ]:
        out = r'\left' + brackets[0] + '\n'

    if len(separate_columns) > 0:

        if latexarraytype != 'array':
            raise ValueError('column separators can only be used for `latexarraytype = "array"`')

        colstr = '{'

        for i in range(ncol):
            if i in separate_columns and i > 0:

                colstr += '|'
            colstr += 'c'

        colstr += '}'
    else:
        colstr = '{}'

    out += r'\begin{' + latexarraytype + '}' +colstr+'\n'

    for i in np.arange(nrow):

        if i in separate_rows and i > 0:
            out += '  \\hline\n'

        out = out + ' '

        for j in np.arange(ncol):
            this_element = ''
            if np.real(a[i, j]) < 0:
                leadstr = ''
            else:
                leadstr = ' '
            if '.' not in fmt.format(a[i, j]):
                dot_space = ' '
            else:
                dot_space = ''
            if np.iscomplexobj(a[i, j]):
                real = math_form(fmt.format(np.real(a[i, j])),
                                 mathform=mathform)
                real = real.lstrip(' ')
                imag = math_form(fmt.format(np.imag(a[i, j])),
                                 is_imaginary=True,
                                 mathform=mathform)
                imag = imag.lstrip(' ')
                if not (imag.startswith('-') or imag.startswith('+')):
                    number = real + '+' + imag
                else:
                    number = real + imag
                this_element = (
                         this_element
                       + leadstr
                       + number
                       + imstring
                       + dot_space
                       )
            else:
                this_element = (
                         this_element
                       + leadstr
                       + math_form(fmt.format(np.real(a[i, j])),
                                   mathform=mathform)
                       + dot_space
                       )

            if (i,j) in mark_elements:
                this_element = r'\colorbox{'+ mark_color +'}{$'+ this_element+'$}  '

            if j < ncol-1:
                this_element += r' & '

            out += this_element

        if i < nrow-1:
            out = out + '\\\\\n'

    out = out + '\n' + r'\end{' + latexarraytype + '}'

    if brackets is not None and latexarraytype not in [
                "bmatrix",
                "pmatrix",
                "vmatrix",
                "Bmatrix",
                "Vmatrix",
            ]:
        out += '\n\\right' + brackets[1]

    return out
