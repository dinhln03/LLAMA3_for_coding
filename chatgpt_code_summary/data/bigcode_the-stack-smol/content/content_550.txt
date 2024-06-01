import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import json
import glob
import os
import argparse
from typing import Tuple, Union, List
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool


pd.options.mode.chained_assignment = None  # default='warn'


# ====================================================================


def get_data(img_pth: Union[str, os.PathLike]) -> dict:
    """Get a single data from the given file.json path"""
    with open(img_pth, 'r') as f:
        data = json.load(f)
    return data


def get_original_df(
        path: Union[str, os.PathLike],
        filename: str,
        processes_per_cpu: int = 2) -> Tuple[pd.DataFrame, bool]:
    """Get a DataFrame from all the can_bus*.json files in the dataset"""
    save_path = os.path.join(os.getcwd(), 'data_analysis', filename)

    if os.path.isfile(save_path):
        print('.npy file exists, loading it...')
        data = list(np.load(save_path, allow_pickle=True))
    else:
        # Construct the dataset
        print('.npy file not found, constructing it...')
        all_data_paths = sorted(glob.glob(os.path.join(path, '**/can_bus*.json'), recursive=True))

        with Pool(os.cpu_count() * processes_per_cpu) as p:
            data = list(tqdm(p.imap(get_data, all_data_paths), total=len(all_data_paths)))

        np.save(save_path, data)

    # Create dataframe with the data
    df = pd.DataFrame(data)
    print(df.describe())

    return df, False

# ====================================================================


def get_augmented_df(preloads_name: str) -> Tuple[pd.DataFrame, bool]:
    """Use the preloads file to load the data; will be augmented, as that's what we did"""
    assert preloads_name.endswith('.npy')
    data = np.load(os.path.join(os.getcwd(), '_preloads', preloads_name), allow_pickle=True)[1]
    df = pd.DataFrame(data)
    print(df.describe())
    return df, True

# ====================================================================


def violin_plot(df: pd.DataFrame, save_name: str, augmented: bool) -> None:
    """Save violin plot for the interesting parameters using df"""
    directions_dict = {'No Action': 2.0, 'Turn Left': 3.0, 'Turn Right': 4.0, 'Continue Straight': 5.0}

    # Auxiliary function for setting the quartile lines
    def set_lines(ax):
        for l in ax.lines:
            l.set_linestyle('--')
            l.set_linewidth(0.6)
            l.set_color('white')
            l.set_alpha(0.7)
        for l in ax.lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1.3)
            l.set_color('black')
            l.set_alpha(0.8)

    for key in directions_dict:
        # Get respective subset of the dataframe
        data = df[df['directions'] == directions_dict[key]]
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(1, 4)

        fig.add_subplot(gs[0, 0])
        ax = sns.violinplot(y='steer', data=data, color='r', inner='quartile')
        set_lines(ax)

        fig.add_subplot(gs[0, 1])
        ax = sns.violinplot(y='throttle', data=data, color='g', inner='quartile')
        set_lines(ax)

        fig.add_subplot(gs[0, 2])
        ax = sns.violinplot(y='brake', data=data, color='b', inner='quartile')
        set_lines(ax)

        fig.add_subplot(gs[0, 3])
        ax = sns.violinplot(y='speed', data=data, color='m', inner='quartile')
        set_lines(ax)

        # When using tight layout, we need the title to be spaced accordingly
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        stitle = f'Direction: {key} - $N={len(data)}$ - ${100 * len(data)/len(df):6.3f}$% of total'
        stitle = f'{stitle} - Augmented' if augmented else stitle
        fig.suptitle(stitle, fontsize=16)

        fname = f'{save_name}-{key.replace(" ", "")}'
        fname = f'{fname}-aug' if augmented else fname
        fig_name = os.path.join(os.getcwd(), 'data_analysis', save_name, 'violin_plots', f'{fname}.png')
        os.makedirs(os.path.join(os.getcwd(), 'data_analysis', save_name, 'violin_plots'), exist_ok=True)
        plt.savefig(fig_name)
        plt.close()


# ====================================================================


def plot_clients(path: Union[str, os.PathLike], df: pd.DataFrame, augmented: bool, speed_factor: float) -> None:
    """Plot the steer, throttle, brake, and speed of a client during its data collection"""
    # Some sanity check
    if path.endswith(os.sep):
        path = path[:-1]

    # Get dataset name and make the necessary directories
    dataset_name = os.path.basename(path)
    s_path = os.path.join(os.getcwd(), 'data_analysis', dataset_name, 'clients')
    os.makedirs(s_path, exist_ok=True)

    # Get the number of clients/cars that collected the data
    clients = glob.glob(os.path.join(path, '**/*'))
    clients = [cl for cl in clients if os.path.isdir(cl)]  # Remove path of metadata.json
    num_clients = len(clients)

    # Total number of frames and for a single client
    num_frames = len(df)
    num_frames_per_client = num_frames // num_clients

    # Aux function
    def get_change_locs(df: pd.DataFrame, cli: int) -> Tuple[List[int], List[float]]:
        """Get the index and directions from the df of the actions taken by the client"""
        df['directions_str'] = df['directions'].astype(str)  # In order to compare, turn directions into a string
        # Shift directions column by 1 (filling the top with the head), and compare to the original
        df['change'] = df['directions_str'].shift(1, fill_value=df['directions_str'].head(1)) != df['directions_str']

        # Get the rows where there's a change
        index_change = list(df.loc[df['change'] == True].index.values)
        # Add the first frame
        index_change = [(cli - 1) * len(df)] + index_change
        # For these indexes, get the value of the direction
        dirs = list(df['directions'][index_change].values)
        # Add the last frame
        index_change = index_change + [cli * len(df) - 1]

        return index_change, dirs

    # Dictionaries containing the name and color for plotting the direction given to the car
    my_labels = {2.0: 'No Action', 3.0: 'Turn Left', 4.0: 'Turn Right', 5.0: 'Continue Straight'}
    colors = {2.0: 'gold', 3.0: 'gray', 4.0: 'cyan', 5.0: 'magenta'}

    # Initialize the total counts per action
    total_action_counts = Counter({2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0})
    max_speed_clients = {}

    idx_change_clients = {}
    dirs_clients = {}

    # Make a plot for each client
    for client in tqdm(range(1, num_clients + 1), total=num_clients, unit='clients'):
        if augmented:
            # Dataframe will have augmented data, which uses center, left, right, center, ... data
            df_client = df[(client - 1) * num_frames_per_client: client * num_frames_per_client: 3]
        else:
            df_client = df[(client - 1) * num_frames_per_client: client * num_frames_per_client]

            # Augmented data will have been normalized already
            df_client['speed'] = df_client['speed'].div(speed_factor)  # normalize to range [0, 1]

        # The actual max speed (see if it differs from collected data)
        actual_max_speed = df_client['speed'].max()
        max_speed_clients[client] = actual_max_speed

        # Build the plot
        fig, ax = plt.subplots(figsize=(48, 16))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        df_client.plot(y=['steer', 'throttle', 'brake', 'speed'], ax=ax)

        # Set the area colors for when an direction is taken
        idx_change, dirs = get_change_locs(df_client, client)
        for idx, dir in enumerate(dirs):
            ax.axvspan(idx_change[idx], idx_change[idx + 1], facecolor=colors[dir], alpha=0.5, label=my_labels[dir])

        # Save these index and directions for each client
        idx_change_clients[f'client_{client:02d}'] = [int(idx) for idx in idx_change]
        dirs_clients[f'client_{client:02d}'] = [float(d) for d in dirs]

        # Count the directions taken by the client
        dirs_count = Counter(dirs)
        # Add this to the total for the whole dataset
        total_action_counts += dirs_count
        # Add the counts to the title
        total_actions = ''
        for key in my_labels:
            total_actions += f' - {my_labels[key]}: {dirs_count[key]}'

        # Set title and x and y axes labels
        suptitle = f'Client {client} - Actual max speed: {actual_max_speed:.4f}'
        suptitle = f'{suptitle} - Augmented' if augmented else suptitle
        suptitle = f'{suptitle}{total_actions}'
        plt.suptitle(suptitle, fontsize=30)
        plt.xlabel('Frame idx', fontsize=22)
        plt.ylabel('Normed value', fontsize=22)
        plt.xticks(list(range((client - 1) * num_frames_per_client,
                              client * num_frames_per_client + 1, len(df_client) // 20)))  # ticks in 5% increments

        # Fix the legend / remove duplicated areas and labels
        hand, labl = ax.get_legend_handles_labels()
        handout = []
        lablout = []
        for h, l in zip(hand, labl):
            if l not in lablout:
                lablout.append(l)
                handout.append(h)

        ax.legend(handout, lablout, fontsize='x-large')
        sname = os.path.join(s_path, f'{dataset_name}_Client{client:02d}')
        sname = f'{sname}-aug' if augmented else sname
        plt.savefig(f'{sname}.png', dpi=300)
        plt.close()

    # Add summary and save it as a JSON file
    actions_summary = {
        'avg_no_action': total_action_counts[2.0] / num_clients,
        'avg_turn_left': total_action_counts[3.0] / num_clients,
        'avg_turn_right': total_action_counts[4.0] / num_clients,
        'avg_continue_straight': total_action_counts[5.0] / num_clients
    }

    summary = {
        'num_clients': num_clients,
        'num_frames_per_client': num_frames_per_client,
        'hours_per_client': num_frames_per_client / (20 * 60 * 60),
        'total_action_counts': total_action_counts,
        'actions_summary': actions_summary,
        'max_speed_clients': max_speed_clients,
        'idx_change_clients': idx_change_clients,
        'dirs_clients': dirs_clients
    }

    with open(os.path.join(s_path, f'{dataset_name}-summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)


# ====================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, help='Path to the head of the dataset', required=True)
    parser.add_argument('--filename', type=str, help='Name of file to save', default=None)
    parser.add_argument('--preloads-name', type=str, help='Name of preload file', default=None)
    parser.add_argument('--processes-per-cpu', '-proc', type=int, help='Processes per cpu (default: %(default)s)', default=2)
    parser.add_argument('--speed-factor', '-sf', type=float, help='Speed factor to normalize data (default: %(default)s)', default=14.0)
    parser.add_argument('--plot-clients', action='store_true', help='Add flag to plot the actions and speed of a client')

    args = parser.parse_args()

    # Create dir if it doesn't exist
    if not os.path.exists(os.path.join(os.getcwd(), 'data_analysis')):
        os.mkdir(os.path.join(os.getcwd(), 'data_analysis'))

    print('Getting the dataframe...')
    if args.preloads_name is not None:
        # Preloaded data is augmented
        df, augmented = get_augmented_df(preloads_name=args.preloads_name)
        save_name = os.path.basename(args.preloads_name).split('.')[0]
    else:
        assert args.filename is not None
        assert args.filename.endswith('.npy')
        df, augmented = get_original_df(args.path, args.filename, args.processes_per_cpu)
        save_name = os.path.basename(args.filename).split('.')[0]

    # Create and save the violin plots
    print('Plotting data...')
    violin_plot(df, save_name, augmented)

    if args.plot_clients:
        print(f'Plotting actions taken by all clients in {args.path}...')
        plot_clients(path=args.path, df=df, augmented=augmented, speed_factor=args.speed_factor)

    print('Done!')


# ====================================================================


if __name__ == '__main__':
    main()


# ====================================================================
