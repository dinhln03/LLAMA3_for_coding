'''Ensemble some predictions. '''
import argparse
import collections
import math
from scipy.special import logsumexp
import sys

MODES = ['mean', 'max', 'logsumexp', 'noisy_or', 'log_noisy_or', 'odds_ratio']

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--weights', '-w', type=lambda x:[float(t) for t in x.split(',')],
                        help='Comma-separated lit of multiplizer per file')
    parser.add_argument('--out-file', '-o', default=None, help='Where to write all output')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args(args)

def read_preds(fn):
    preds = []
    with open(fn) as f:
        for line in f:
            idx, pmid, drug, gene, variant, prob = line.strip().split('\t')
            prob = float(prob)
            preds.append((pmid, drug, gene, variant, prob))

    return preds

def main(OPTS):
    preds_all = [read_preds(fn) for fn in OPTS.files]
    groups = collections.defaultdict(list)
    for i, preds in enumerate(preds_all):
        if OPTS.weights:
            weight = OPTS.weights[i]
        else:
            weight = 1.0
        for pmid, drug, gene, variant, prob in preds:
            groups[(pmid, drug, gene, variant)].append(weight * prob)

    results = []
    for i , ((pmid, drug, gene, variant), prob_list) in enumerate(groups.items()):
        if OPTS.mode == 'mean':
            prob = sum(prob_list) / len(prob_list)
        elif OPTS.mode == 'max':
            prob = max(prob_list)
        elif OPTS.mode == 'logsumexp':
            prob = logsumexp(prob_list)
        elif OPTS.mode == 'noisy_or':
            prob_no_rel = 1.0
            for p in prob_list:
                prob_no_rel *= 1.0 - p
            prob =1.0 - prob_no_rel
        elif OPTS.mode == 'log_noisy_or':
            log_prob_no_rel = 0.0
            for p in prob_list:
                if p < 1.0:
                    log_prob_no_rel += math.log(1.0 - p)
                else:
                    log_prob_no_rel -= 1000000
            prob = -log_prob_no_rel
        elif OPTS.mode == 'odds_ratio':
            cur_log_odds = 0.0
            for p in prob_list:
                cur_log_odds += 10  + 0.001 * p #math.log(p / (1.0 - p) * 100000000)
            prob = cur_log_odds
        else:
            raise ValueError(OPTS.mode)
        results.append((i, pmid, drug, gene, variant, prob))

    with open(OPTS.out_file, 'w') as f:
        for item in results:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(*item))

if __name__ == '__main__':
    OPTS = parse_args(sys.argv[1:])
    main(OPTS)






