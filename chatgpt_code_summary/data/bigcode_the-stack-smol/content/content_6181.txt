"""
Toxopy (https://github.com/bchaselab/Toxopy)
© M. Alyetama, University of Nebraska at Omaha
Licensed under the terms of the MIT license
"""

from toxopy import fwarnings, trials
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def dlcboxplot(file,
               variable,
               ylab,
               comparison,
               jitter=False,
               colors=False,
               title=False,
               save=False,
               output_dir=None):
    """
  file is typically 'dlc_all_avgs_updated.csv'
  variable is either 'cat_ditance' or 'vel'
  ylab is the y-axis label
  colors is a list of two colors (e.g., ["#0062FF", "#DB62FF"])
  output_dir to save the plot in a specific dir when save is True
  """

    df = pd.read_csv(file)
    tls = trials()
    new = ['FT', 'ALONE1', 'SALINE1', 'ALONE2', 'URINE1',
           'ALONE3', 'SALINE2', 'ALONE4', 'URINE2', 'ALONE5']
    if variable == 'distance':
        df = df[(df['trial'].isin(tls[0::2]))]
    d = {}

    for i, j in zip(new, tls):
        d[j] = i

    df = df.replace(d)
    df = df[df['var'] == variable]

    sns.set(style='ticks', font_scale=1)

    plt.figure(figsize=(13, 5), dpi=100)

    if comparison == 'infection_status':
        test, control = 'Infected', 'Control'
        comparing = 'infection_status'
        legend = 'Infection Status'
    elif comparison == 'indoor_outdoor_status':
        test, control = 'Indoor-outdoor', 'Indoor'
        comparing = 'indoor_outdoor_status'
        legend = 'Indoor-outdoor Status'

    if colors is False:
        my_pal = {control: '#00FFFF', test: '#E60E3C'}
    else:
        my_pal = {control: colors[0], test: colors[1]}

    ax = sns.boxplot(x='trial',
                     y='value',
                     data=df,
                     hue=comparing,
                     palette=my_pal)

    if jitter is True:
        sns.stripplot(x='trial',
                      y='value',
                      data=df,
                      color='black',
                      size=3,
                      jitter=1)

    if variable != 'distance':
        for i in range(len(df['trial'].unique())-1):
            if variable == 'vel':
                plt.vlines(i+.5, 10, 45, linestyles='solid',
                           colors='black', alpha=0.2)
            elif variable == 'cat_distance':
                plt.vlines(i+.5, 0, 1.3, linestyles='solid',
                           colors='black', alpha=0.2)

    if title is not False:
        plt.title(title, fontsize=12)
    else:
        pass

    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)

    ax.legend(title=legend)

    plt.legend(title=legend)
    '''add significance bars and asterisks between boxes.
    [first pair, second pair], ..., [|, –], ...'''
    if variable == 'vel':
        l = [[7.75, 5.75], [8.25, 6.25], [26, 28], [31, 33]]
    elif variable == 'cat_distance':
        l = [[7.75, 5.75], [8.25, 6.25], [0.85, 0.9], [0.95, 1]]

    for x1, x2, y1, y2 in zip(l[0], l[1], l[2], l[3]):
        sig = plt.plot([x1, x1, x2, x2], [y1, y2, y2, y1],
                       linewidth=1,
                       color='k')
        plt.text((x1 + x2) * .5, y2 + 0, "*",
                 ha='center', va='bottom', fontsize=18)

    plt.show()

    fig = ax.get_figure()

    if save is True:

        def sav(myString):
            return fig.savefig(myString,
                               bbox_inches='tight',
                               dpi=100,
                               pad_inches=0.1)

            if output_dir is not None:
                sav(f'{output_dir}/{variable}.png')
            else:
                sav(f'{variable}.png')
