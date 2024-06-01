# +
import numpy as np
import holoviews as hv
from holoviews import opts
import matplotlib.pyplot as plt

from plotsun import plot_sun

hv.extension('bokeh', 'matplotlib')
# -

# # Load data

data = np.load('npz_timeseries/subset.npz')
arr = data['arr']
stack = data['stack']
sun = data['sun']
print(arr.shape, stack.shape, sun.shape)

stack[:,:,25]

plt.imshow(stack[:,:,25], cmap='binary')

# +
stack = hv.Dataset((np.arange(stack.shape[2]),
                    np.arange(stack.shape[0]), 
                    np.arange(stack.shape[1]), 
                    stack),
                    ['Time', 'x', 'y'], 'Shadows')

stack
# -

arr = hv.Dataset((np.arange(arr.shape[0]),
                  np.arange(arr.shape[1]), 
                  arr),
                  ['x', 'y'], 'Elevation')
arr

# # View

opts.defaults(
    opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
    opts.Image(cmap='viridis', invert_yaxis=True, width=400, height=400),
    opts.Labels(text_color='white', text_font_size='8pt', 
                text_align='left', text_baseline='bottom'),
    opts.Path(color='white'),
    opts.Spread(width=600),
    opts.Overlay(show_legend=False))

elevation = arr.to(hv.Image, ['x', 'y'])

shadows = stack.to(hv.Image, ['x', 'y'])

elevation

dims = {'figsize':(4,5), 'top':1, 'bottom':0, 'left':0.2, 'right':0.95}
plot_sun(sunposition=sun, d=dims)

elevation * shadows

stack[:,:,24]


