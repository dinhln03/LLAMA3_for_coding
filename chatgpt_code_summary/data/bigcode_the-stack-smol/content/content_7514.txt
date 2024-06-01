import argparse


def train_args():
    """
        Retrieves and parses the 3 command line arguments provided by the user when
        they run the program from a terminal window. This function uses Python's
        argparse module to created and defined these 3 command line arguments. If
        the user fails to provide some or all of the 3 arguments, then the default
        values are used for the missing arguments.

        :return: results: data structure that stores the command line arguments object
     """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', action='store',
                        dest='data_dir',
                        default='flowers',
                        help='Path the directory of the dataset, should contain sub-directories /train, /test, /valid')

    parser.add_argument('--save_dir', action='store',
                        dest='save_dir',
                        default='checkpoint.pth',
                        help='Set directory to save checkpoints')

    parser.add_argument('--arch', action='store',
                        dest='arch',
                        default='vgg16',
                        help='Choose architecture. Default: vgg16')

    parser.add_argument('--learning_rate', action='store',
                        dest='learning_rate',
                        default=0.003,
                        help='Set the learning rate',
                        type=float)

    parser.add_argument('--hidden_units', action='store',
                        dest='hidden_units',
                        default=256,
                        help='Add the hidden units',
                        type=int)

    parser.add_argument('--epochs', action='store',
                        dest='epochs',
                        default=30,
                        help='Add number of epoch cycles',
                        type=int)
    parser.add_argument('--gpu', action='store_true',
                        dest='gpu',
                        help='Activate GPU')

    results = parser.parse_args()

    return results
