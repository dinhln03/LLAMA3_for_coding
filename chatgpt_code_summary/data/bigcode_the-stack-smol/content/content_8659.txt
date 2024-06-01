import argparse
import pandas as pd
import re

ass15 = 'assist15', 'ASSISTments 2015'
ass17 = 'assist17', 'ASSISTments 2017'
prog19 = 'prog19', 'Programming 2019'
synth_k2 = 'synth-k2', 'Synthetic-K2'
synth_k5 = 'synth-k5', 'Synthetic-K5'
ass09up = 'assist09up', 'ASSISTments 2009 Updated'
stat = 'stat', 'Statics'
intro_prog = 'intro-prog', 'IntroProg'


def sl_dict(a, b):
    return {'short': a, 'long': b}


dataset_tups = ass15, ass17, prog19, synth_k2, synth_k5, ass09up, stat, intro_prog
datasets = {**{s: sl_dict(s, l) for s, l in dataset_tups}, **{l: sl_dict(s, l) for s, l in dataset_tups}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Yay, I\'m a description!',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('kfold_results_filename')

    parser.add_argument('--min', action='store_true')
    parser.add_argument('--metric',
                        default='auc',
                        choices={'acc', 'auc', 'prec', 'recall', 'f1', 'mcc', 'rmse', 'aic', 'aicc', 'bic'})
    args = parser.parse_args()

    kfold_results = pd.read_csv(args.kfold_results_filename, encoding='latin')
    kfold_results.dataset = kfold_results.dataset.apply(lambda x: datasets[x]['long'])
    kfold_results.dataset = kfold_results.dataset.apply(lambda x: datasets[x]['long'] if x in datasets else x)

    max_filter_col = f'{args.metric}-sd'
    kfold_results[max_filter_col] = kfold_results[args.metric].apply(lambda x: float(re.split(r'[^0-9.]', x)[0]))

    kfold_max_results = kfold_results.loc[kfold_results.groupby(['dataset', 'model'])[max_filter_col].idxmax()] \
        if not args.min \
        else kfold_results.loc[kfold_results.groupby(['dataset', 'model'])[max_filter_col].idxmin()]

    best = 'min' if args.metric in ('rmse', 'aic', 'aicc', 'bic') else 'max'
    if 'fold-results' in args.kfold_results_filename:
        output_filename = args.kfold_results_filename.replace("-results", f"-{best}-{args.metric}-results")
    else:
        output_filename = f'kfold-results-{best}-{args.metric}-results.csv'

    print(f'wrote {output_filename}')
    kfold_max_results = kfold_max_results.drop([max_filter_col], axis=1).sort_values(
        by=['dataset', args.metric], ascending=False)
    kfold_max_results.to_csv(output_filename, index=False)
