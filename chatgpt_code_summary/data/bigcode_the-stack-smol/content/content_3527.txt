import json
import os
import argparse


def main(split):
    with open(args.data_path + '/' + split + '.json') as f:
        data = json.load(f)
    sparc = []
    for i in range(len(data)):
        d = data[i]
        for j in range(len(d['interaction'])):
            turn = d['interaction'][j]
            sparc.append({})
            sparc[-1]['interaction_id'] = i + 1
            sparc[-1]['turn_id'] = j + 1
            sparc[-1]['db_id'] = d['database_id']
            sparc[-1]['query'] = turn['query']
            sparc[-1]['question'] = turn['utterance'].replace('“', '\"').replace(
                '”', '\"').replace('‘', '\"').replace('’', '\"') + '>>>'
            sparc[-1]['query_toks_no_value'] = turn['query_toks_no_value']
            sparc[-1]['question_toks'] = turn['utterance_toks']
            if j:
                sparc[-1]['question'] = sparc[-1]['question'] + \
                    sparc[-2]['question']
            sparc[-1]['sql'] = turn['sql']

            sparc[-1]['question'] = sparc[-1]['question'].replace('*', '')
            sparc[-1]['question_toks'] = [tok.replace('*', '')
                                          for tok in sparc[-1]['question_toks'] if tok != '*']
    with open(os.path.join(args.data_path, split) + '.json', 'w') as f:
        json.dump(sparc, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", '-dp', type=str)
    args = parser.parse_args()

    for split in ['train', 'dev']:
        main(split)

    print('convert done')
