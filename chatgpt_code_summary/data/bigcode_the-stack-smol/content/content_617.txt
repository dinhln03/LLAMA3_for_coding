import os
import pandas as pd
import re


def sort_human(l):
    """Sort a list of strings by numerical."""
    def convert(text): return float(text) if text.isdigit() else text

    def alphanum(key): return [convert(c)
                               for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l


def data_merge_by_batch(parent_directory, verbose=True):
    """Merge a set of parameters.csv files into one.

    This is intended for use with batch processes from Legion, with each batch
    being 1000 runs longand numbered with integer values.

    Parameters
    ----------
    parent_directory : :obj:`list` of :obj:`str`
        Parent directory to a set of directories each containing model runs and
        a parameters.csv file.
    verbose : :obj:`boolean`, optional
        Boolean indicator of whether to print extra information.

    Returns
    -------
    None
        Concatenated will be written to file in `parent_directory`

    """
    dirs = [os.path.abspath(os.path.join(parent_directory, d))
            for d in os.listdir(parent_directory)
            if os.path.isdir(os.path.abspath(
                os.path.join(parent_directory, d))) and d != 'archives']
    dirs = sort_human(dirs)
    if verbose:
        print(dirs)
    dfs = []
    for d in dirs:
        try:
            dfs.append(pd.read_csv(os.path.join(d, 'parameters.csv')))
            ii = len(dfs) - 1
            print("Processing parameter file {}".format(ii))
            if ii is not 0:
                dfs[ii]['ix'] = dfs[ii].index.values + \
                    dfs[ii - 1]['ix'].values[-1] + 1
            else:
                dfs[ii]['ix'] = dfs[ii].index.values

            if os.path.split(d)[1].split('_')[-1].isdigit():
                print(os.path.split(d)[1].split('_')[-1])
                dfs[ii]['Batch'] = int(os.path.split(d)[1].split('_')[-1])
            else:
                print("Batch number not found for {}".format(d))
                continue
        except FileNotFoundError:
            print("No parameters file in {}".format(d))
            continue
    if verbose:
        print("{} dataframes  to be joined".format(len(dfs)))
    # for ii in range(len(dfs)):
        # if ii is not 0:
        #     dfs[ii]['ix'] = dfs[ii].index.values + dfs[ii - 1]['ix'].values[-1]
        # else:
        #     dfs[ii]['ix'] = dfs[ii].index.values
        # if os.path.split(dirs[ii])[1][:4].isdigit():
        #     print(os.path.split(dirs[ii])[1][:4])
        #     dfs[ii]['Start Time'] = os.path.split(dirs[ii])[1][:4]
        # else:
        #     continue
    df = pd.concat(dfs)
    df.index = range(len(df))
    output_file = os.path.join(parent_directory,
                               'all_parameters.csv')
    df.to_csv(output_file, index=False)

    return output_file
