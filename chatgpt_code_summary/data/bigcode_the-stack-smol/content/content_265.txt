import argparse
import sys, os
import logging
from utils.misc import ArgParseDefault, add_bool_arg

USAGE_DESC = """
python main.py <command> [<args>]

Available commands:
  init             Initialize project
  sync             Sync project data from S3
  parse            Preprocessing of data to generate `/data/1_parsed`
  sample           Sample cleaned data to generate `data/2_sampled`
  batch            Creates a new batch of tweets from a sampled file in `/data/2_sampled`
  clean_labels     Clean labels generated from (`data/3_labelled`) and merge/clean to generate `/data/4_cleaned_labels`
  stats            Output various stats about project
  split            Splits data into training, dev and test data
  prepare_predict  Prepares parsed data for prediction with txcl
"""

STATS_USAGE_DESC = """
python main.py stats <command> [<args>]

Available commands:
  all                 Run all
  overview            Show overview
  sample              Show sampling stats
  annotation          Show annotation summary
  annotation_cleaned  Show cleaned annotation summary
  annotator_outliers  Show annotator outliers
"""

class ArgParse(object):
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
        parser = ArgParseDefault(
                description='',
                usage=USAGE_DESC)
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            sys.exit(1)
        getattr(self, args.command)()

    def init(self):
        from utils.task_helpers import init
        parser = ArgParseDefault(description='Initialize project')
        parser.add_argument('-p', '--project', type=str, required=False, default='', dest='project', help='Name of project to initialize')
        parser.add_argument('--template', dest='template', action='store_true', default=False, help='Initialize project manually.')
        args = parser.parse_args(sys.argv[2:])
        init(args.project, args.template)

    def sync(self):
        from utils.task_helpers import sync
        parser = ArgParseDefault(description='Sync project data from S3')
        parser.add_argument('-s', '--source', choices=['all', 'streaming', 'annotation', 'media'], required=False, default='all', help='Type of data to be synced. By default sync all data belonging to this project.')
        parser.add_argument('-l', '--last', required=False, type=int, help='Sync streaming data of last n days')
        args = parser.parse_args(sys.argv[2:])
        sync(data_type=args.source, last_n_days=args.last)

    def parse(self):
        import utils.processing.parse_tweets as parse_tweets
        parser = ArgParseDefault(description='Preprocess raw data to create parquet files in `data/1_parsed`')
        parser.add_argument('--no-parallel', dest='no_parallel', action='store_true', default=False, help='Do not run in parallel')
        parser.add_argument('--extend', dest='extend', action='store_true', default=False, help='Extend existing parsed data')
        parser.add_argument('--ray_num_cpus', type=int, default=None, help='Limit the number of worker processes for Ray during the memory intensive merge phase (by default using maximum worker processes)')
        add_bool_arg(parser, 'extract_retweets', default=True, help='Extract top-level retweets')
        add_bool_arg(parser, 'extract_quotes', default=True, help='Extract top-level quotes')
        add_bool_arg(parser, 'omit_last_day', default=True, help='Omit parsing data from the last day')
        args = parser.parse_args(sys.argv[2:])
        parse_tweets.run(no_parallel=args.no_parallel, extract_retweets=args.extract_retweets, extract_quotes=args.extract_quotes, extend=args.extend, omit_last_day=args.omit_last_day, ray_num_cpus=args.ray_num_cpus)

    def sample(self):
        import utils.processing.sample_tweets as sample_tweets
        parser = ArgParseDefault(description='Sample cleaned data to generate `data/2_sampled`')
        parser.add_argument('-s', '--size', type=int, required=True, dest='size', help='Number of tweets to sample')
        parser.add_argument('-bs', '--bin_size', type=int, required=False, help='Number of tweets per bin')
        parser.add_argument('-m', '--mode', choices=['monthly', 'random'], required=False, default='random', help='Sampling mode. Random: Sample randomly. Monthly: Try to sample evenly within months.')
        parser.add_argument('-l', '--langs', default=[], nargs='+', required=False, help='Filter by language(s)')
        parser.add_argument('--contains_keywords', default=False, action='store_true', help='Only sample from tweets which include keywords')
        parser.add_argument('--min_token_count', default=3, type=int, required=False, help='Minimum number of tokens')
        parser.add_argument('--include_replies', default=False, action='store_true', help='Include replies')
        parser.add_argument('--seed', type=int, required=False, default=None, help='Random state split')
        parser.add_argument('--extend', action='store_true', help='Extending existing sample given by seed by removing already labelled tweets. If size is <= original sample size this has no effect except removing labelled tweets');
        add_bool_arg(parser, 'anonymize', default=True, help='Replace usernames and URLs with filler (@user and <url>)')
        parser.add_argument('--max_date', required=False, default=None, help='Sample until date (YYYY-MM-DD), default: No max')
        parser.add_argument('--min_date', required=False, default=None, help='Sample from date (YYYY-MM-DD), default: No min')
        args = parser.parse_args(sys.argv[2:])
        sample_tweets.run(size=args.size, contains_keywords=args.contains_keywords, anonymize=args.anonymize, min_token_count=args.min_token_count, langs=args.langs, include_replies=args.include_replies, mode=args.mode, seed=args.seed, extend=args.extend, bin_size=args.bin_size, min_date=args.min_date, max_date=args.max_date)

    def batch(self):
        from utils.processing.sample_tweets import SampleGenerator
        parser = ArgParseDefault(description='Generate new batch for labelling. As a result a new csv will be created in `data/2_sampled/batch_{batch_id}/`')
        parser.add_argument('-N', '--num_tweets', type=int, default=None, help='The number of tweets to be generated in new batch')
        parser.add_argument('-b', '--batch', type=int, default=None, help='The batch id to be generated, default: Automatically find next batch')
        parser.add_argument('--ignore-previous', dest='ignore_previous', action='store_true', default=False, help='Also sample tweets from old batches which were not annotated')
        parser.add_argument('--stats-only', dest='stats_only', action='store_true', default=False, help='Show stats only')
        args = parser.parse_args(sys.argv[2:])
        s = SampleGenerator()
        if args.stats_only:
            s.stats(ignore_previous=args.ignore_previous)
        else:
            s.generate_batch(num_tweets=args.num_tweets, batch_id=args.batch, ignore_previous=args.ignore_previous)

    def clean_labels(self):
        import utils.processing.clean_labels as clean_labels
        parser = ArgParseDefault(description='Clean/merge labels from different batches to generate final training input')
        parser.add_argument('-s', '--selection-criterion', dest='selection_criterion', choices=['majority', 'unanimous'], required=False, default='majority', help='Can be "majority" (use majority vote) or "unanimous" (only select tweets with perfect agreement)')
        parser.add_argument('-l', '--min-labels-cutoff', dest='min_labels_cutoff', type=int, required=False, default=3, help='Discard all tweets having less than min_labels_cutoff annotations')
        parser.add_argument('-a', '--selection-agreement', dest='selection_agreement', type=float, required=False, default=None, help='Consider only tweets with a certain level of annotation agreement. If provided overwrites selection_criterion param.')
        parser.add_argument('-m', '--mode', choices=['mturk', 'local', 'public', 'other', 'all'], type=str, required=False, default='all', help='Annotation mode which was used. Can be `mturk`, `local`, `public`, `other` or `all`')
        parser.add_argument('--is-relevant', dest='is_relevant', action='store_true', help='Filter tweets which have been annotated as relevant/related')
        parser.add_argument('--exclude-incorrect', dest='exclude_incorrect', action='store_true', help='Remove annotations which have been manually flagged as incorrect')
        parser.add_argument('--cutoff-worker-outliers', dest='cutoff_worker_outliers', type=float, default=None, help='Remove all annotations by workers who have agreement scores below certain Z-score threshold (a reasonable value would be 2 or 3)')
        parser.add_argument('--allow-nan', dest='allow_nan', nargs='+', choices=['id', 'text', 'question_id', 'answer_id'], default=[], required=False, help='Allow certain fields to be NaN/empty (by default each annotation has to have the fields id, text, answer_id and question_id)')
        parser.add_argument('--contains-keywords', dest='contains_keywords', default=False, action='store_true', help='Remove annotations in which text does not contain keywords')
        parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output')
        args = parser.parse_args(sys.argv[2:])
        clean_labels.run_clean_labels(args.selection_criterion, args.min_labels_cutoff, args.selection_agreement, args.mode, args.is_relevant, args.exclude_incorrect, args.cutoff_worker_outliers, args.allow_nan, args.contains_keywords, args.verbose)

    def stats(self):
        from utils.task_helpers import stats
        parser = ArgParseDefault(description='Output various stats about project', usage=STATS_USAGE_DESC)
        parser.add_argument('command', choices=['all', 'overview', 'sample', 'annotation', 'annotator_outliers', 'annotation_cleaned'], help='Subcommand to run')
        args = parser.parse_args(sys.argv[2:3])
        if args.command == 'annotation':
            parser = ArgParseDefault(description='Print stats about annotations')
            parser.add_argument('-m', '--mode', choices=['all', 'mturk', 'local', 'public', 'other', '*'], type=str, required=False, default='all', help='Print stats for certain annotation modes only.')
            args = parser.parse_args(sys.argv[3:])
            stats('annotation', **vars(args))
        elif args.command == 'annotator_outliers':
            parser = ArgParseDefault(description='Find annotators which have under-performed compared to others')
            parser.add_argument('-m', '--mode', choices=['mturk', 'local', 'public', 'other'], type=str, required=False, default='mturk', help='Print stats for certain annotation modes only.')
            parser.add_argument('-b', '--batch-name', type=str, required=False, dest='batch_name', default='*', help='Only analyse for specific local/mturk batch name (this looks for a pattern in filename). Default: All data')
            parser.add_argument('--agreement-cutoff', dest='agreement_cutoff', type=float, required=False, default=3, help='Z-value cutoff for inter-worker agreement deviation')
            parser.add_argument('--time-cutoff', dest='time_cutoff', type=float, required=False, default=3, help='Z-value cutoff for average task duration per worker')
            parser.add_argument('--min-tasks', dest='min_tasks', type=int, required=False, default=3, help='Min tasks for worker to have completed before considered as outlier')
            parser.add_argument('--min-comparisons-count', dest='min_comparisons_count', type=int, required=False, default=20, help='Min number of questions to compare for a worker needed to compute agreement score')
            args = parser.parse_args(sys.argv[3:])
            stats('annotator_outliers', **vars(args))
        else:
            stats(args.command)

    def split(self):
        from utils.task_helpers import train_dev_test_split
        parser = ArgParseDefault(description='Split annotated data into training and test data set')
        parser.add_argument('--question', type=str, required=False, default='sentiment', help='Which data to load (has to be a valid question tag)')
        parser.add_argument('--name', type=str, required=False, default='', help='In case there are multiple cleaned labelled data output files give name of file (without csv ending), default: No name provided (works only if a single file is present).')
        parser.add_argument('--balanced-labels', dest='balanced_labels', action='store_true', default=False, help='Ensure equal label balance')
        parser.add_argument('--all-questions', dest='all_questions', action='store_true', default=False, help='Generate files for all available question tags. This overwrites the `question` argument. Default: False.')
        parser.add_argument('--label-tags', dest='label_tags', required=False, default=[], nargs='+', help='Only select examples with certain label tags')
        parser.add_argument('--has-label', dest='has_label', required=False, default='', help='Only select examples which have also been tagged with certain label')
        parser.add_argument('--dev-size', dest='dev_size', type=float, required=False, default=0.2, help='Fraction of dev size')
        parser.add_argument('--test-size', dest='test_size', type=float, required=False, default=0.2, help='Fraction of test size')
        parser.add_argument('--seed', type=int, required=False, default=42, help='Random state split')
        args = parser.parse_args(sys.argv[2:])
        train_dev_test_split(question=args.question, dev_size=args.dev_size, test_size=args.test_size, seed=args.seed, name=args.name, balanced_labels=args.balanced_labels, all_questions=args.all_questions, label_tags=args.label_tags, has_label=args.has_label)
    
    def prepare_predict(self):
        from utils.task_helpers import prepare_predict
        parser = ArgParseDefault(description='Prepare data for prediction with the text-classification library. \
                This function generates two files (1 for text 1 for IDs/created_at) under data/other. The text.csv file can then be predicted.')
        parser.add_argument('--start_date', required=False, default=None, help='Filter start date')
        parser.add_argument('--end_date', required=False, default=None, help='Filter end date')
        add_bool_arg(parser, 'anonymize', default=True, help='Replace usernames and URLs with filler (@user and <url>)')
        parser.add_argument('--url_filler', required=False, default='<url>', help='Filler for urls (if anonymize)')
        parser.add_argument('--user_filler', required=False, default='@user', help='Filler for user names (if anonymize)')
        args = parser.parse_args(sys.argv[2:])
        prepare_predict(args)


if __name__ == '__main__':
    ArgParse()
