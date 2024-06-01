#!/usr/bin/env python3

'''
Edit gizmo snapshot files: compress, delete, transfer across machines.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import os
import sys
import glob
import numpy as np
# local ----
import utilities as ut
from gizmo_analysis import gizmo_io

# default subset of snapshots (65 snapshots)
snapshot_indices_keep = [
    0,  # z = 99
    20, 26, 33, 41, 52,  # z = 10 - 6
    55, 57, 60, 64, 67,  # z = 5.8 - 5.0
    71, 75, 79, 83, 88,  # z = 4.8 - 4.0
    91, 93, 96, 99, 102, 105, 109, 112, 116, 120,  # z = 3.9 - 3.0
    124, 128, 133, 137, 142, 148, 153, 159, 165, 172,  # z = 2.9 - 2.0
    179, 187, 195, 204, 214, 225, 236, 248, 262, 277,  # z = 1.9 - 1.0
    294, 312, 332, 356, 382, 412, 446, 486, 534,  # z = 0.9 - 0.1
    539, 544, 550, 555, 561, 567, 573, 579, 585,  # z = 0.09 - 0.01
    600
]


#===================================================================================================
# compress files
#===================================================================================================
class CompressClass(ut.io.SayClass):

    def compress_snapshots(
        self, directory='output', directory_out='', snapshot_index_limits=[0, 600],
        thread_number=1):
        '''
        Compress all snapshots in input directory.

        Parameters
        ----------
        directory : str : directory of snapshots
        directory_out : str : directory to write compressed snapshots
        snapshot_index_limits : list : min and max snapshot indices to compress
        syncronize : bool : whether to synchronize parallel tasks,
            wait for each thread bundle to complete before starting new bundle
        '''
        snapshot_indices = np.arange(snapshot_index_limits[0], snapshot_index_limits[1] + 1)

        args_list = [(directory, directory_out, snapshot_index)
                     for snapshot_index in snapshot_indices]

        ut.io.run_in_parallel(self.compress_snapshot, args_list, thread_number=thread_number)

    def compress_snapshot(
        self, directory='output', directory_out='', snapshot_index=600,
        analysis_directory='~/analysis', python_executable='python3'):
        '''
        Compress single snapshot (which may be multiple files) in input directory.

        Parameters
        ----------
        directory : str : directory of snapshot
        directory_out : str : directory to write compressed snapshot
        snapshot_index : int : index of snapshot
        analysis_directory : str : directory of analysis code
        '''
        executable = '{} {}/manipulate_hdf5/compactify_hdf5.py -L 0'.format(
            python_executable, analysis_directory)
        snapshot_name_base = 'snap*_{:03d}*'

        if directory[-1] != '/':
            directory += '/'
        if directory_out and directory_out[-1] != '/':
            directory_out += '/'

        path_file_names = glob.glob(directory + snapshot_name_base.format(snapshot_index))

        if len(path_file_names):
            if 'snapdir' in path_file_names[0]:
                path_file_names = glob.glob(path_file_names[0] + '/*')

            path_file_names.sort()

            for path_file_name in path_file_names:
                if directory_out:
                    path_file_name_out = path_file_name.replace(directory, directory_out)
                else:
                    path_file_name_out = path_file_name

                executable_i = '{} -o {} {}'.format(executable, path_file_name_out, path_file_name)
                self.say('executing:  {}'.format(executable_i))
                os.system(executable_i)

    def test_compression(
        self, snapshot_indices='all', simulation_directory='.', snapshot_directory='output',
        compression_level=0):
        '''
        Read headers from all snapshot files in simulation_directory to check whether files have
        been compressed.
        '''
        header_compression_name = 'compression.level'

        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = ut.io.get_path(snapshot_directory)

        Read = gizmo_io.ReadClass()

        compression_wrong_snapshots = []
        compression_none_snapshots = []

        if snapshot_indices is None or snapshot_indices == 'all':
            _path_file_names, snapshot_indices = Read.get_snapshot_file_names_indices(
                simulation_directory + snapshot_directory)
        elif np.isscalar(snapshot_indices):
            snapshot_indices = [snapshot_indices]

        for snapshot_index in snapshot_indices:
            header = Read.read_header('index', snapshot_index, simulation_directory, verbose=False)
            if header_compression_name in header:
                if (compression_level is not None and
                        header[header_compression_name] != compression_level):
                    compression_wrong_snapshots.append(snapshot_index)
            else:
                compression_none_snapshots.append(snapshot_index)

        self.say('* tested {} snapshots: {} - {}'.format(
            len(snapshot_indices), min(snapshot_indices), max(snapshot_indices)))
        self.say('* {} are uncompressed'.format(len(compression_none_snapshots)))
        if len(compression_none_snapshots):
            self.say('{}'.format(compression_none_snapshots))
        self.say('* {} have wrong compression (level != {})'.format(
            len(compression_wrong_snapshots), compression_level))
        if len(compression_wrong_snapshots):
            self.say('{}'.format(compression_wrong_snapshots))


Compress = CompressClass()


#===================================================================================================
# transfer files via globus
#===================================================================================================
class GlobusClass(ut.io.SayClass):

    def submit_transfer(
        self, simulation_path_directory='.', snapshot_directory='output',
        batch_file_name='globus_batch.txt', machine_name='peloton'):
        '''
        Submit globus transfer of simulation files.
        Must initiate from Stampede.

        Parameters
        ----------
        simulation_path_directory : str : '.' or full path + directory of simulation
        snapshot_directory : str : directory of snapshot files within simulation_directory
        batch_file_name : str : name of file to write
        machine_name : str : name of machine transfering files to
        '''
        # set directory from which to transfer
        simulation_path_directory = ut.io.get_path(simulation_path_directory)
        if simulation_path_directory == './':
            simulation_path_directory = os.getcwd()
        if simulation_path_directory[-1] != '/':
            simulation_path_directory += '/'

        command = 'globus transfer $(globus bookmark show stampede){}'.format(
            simulation_path_directory[1:])  # preceeding '/' already in globus bookmark

        path_directories = simulation_path_directory.split('/')
        simulation_directory = path_directories[-2]

        # parse machine + directory to transfer to
        if machine_name == 'peloton':
            if 'elvis' in simulation_directory:
                directory_to = 'm12_elvis'
            else:
                directory_to = simulation_directory.split('_')[0]
            directory_to += '/' + simulation_directory + '/'

            command += ' $(globus bookmark show peloton-scratch){}'.format(directory_to)

        # set globus parameters
        command += ' --sync-level=checksum --preserve-mtime --verify-checksum'
        command += ' --label "{}" --batch < {}'.format(simulation_directory, batch_file_name)

        # write globus batch file
        self.write_batch_file(simulation_path_directory, snapshot_directory, batch_file_name)

        self.say('* executing:\n{}\n'.format(command))
        os.system(command)

    def write_batch_file(
        self, simulation_directory='.', snapshot_directory='output', file_name='globus_batch.txt'):
        '''
        Write batch file that sets files to transfer via globus.

        Parameters
        ----------
        simulation_directory : str : directory of simulation
        snapshot_directory : str : directory of snapshot files within simulation_directory
        file_name : str : name of batch file to write
        '''
        simulation_directory = ut.io.get_path(simulation_directory)
        snapshot_directory = ut.io.get_path(snapshot_directory)

        transfer_string = ''

        # general files
        transfer_items = [
            'gizmo/',
            'gizmo_config.sh',
            'gizmo_parameters.txt',
            'gizmo_parameters.txt-usedvalues',
            'gizmo.out.txt',
            'snapshot_times.txt',
            'notes.txt',

            'track/',
            'halo/rockstar_dm/catalog_hdf5/',
        ]
        for transfer_item in transfer_items:
            if os.path.exists(simulation_directory + transfer_item):
                command = '{} {}'
                if transfer_item[-1] == '/':
                    transfer_item = transfer_item[:-1]
                    command += ' --recursive'
                command = command.format(transfer_item, transfer_item) + '\n'
                transfer_string += command

        # initial condition files
        transfer_items = glob.glob(simulation_directory + 'initial_condition*/*')
        for transfer_item in transfer_items:
            if '.ics' not in transfer_item:
                transfer_item = transfer_item.replace(simulation_directory, '')
                command = '{} {}\n'.format(transfer_item, transfer_item)
                transfer_string += command

        # snapshot files
        for snapshot_index in snapshot_indices_keep:
            snapshot_name = '{}snapdir_{:03d}'.format(snapshot_directory, snapshot_index)
            if os.path.exists(simulation_directory + snapshot_name):
                snapshot_string = '{} {} --recursive\n'.format(snapshot_name, snapshot_name)
                transfer_string += snapshot_string

            snapshot_name = '{}snapshot_{:03d}.hdf5'.format(snapshot_directory, snapshot_index)
            if os.path.exists(simulation_directory + snapshot_name):
                snapshot_string = '{} {}\n'.format(snapshot_name, snapshot_name)
                transfer_string += snapshot_string

        with open(file_name, 'w') as file_out:
            file_out.write(transfer_string)


Globus = GlobusClass()


#===================================================================================================
# transfer files via rsync
#===================================================================================================
def rsync_snapshots(
    machine_name, simulation_directory_from='', simulation_directory_to='.',
    snapshot_indices=snapshot_indices_keep):
    '''
    Use rsync to copy snapshot file[s].

    Parameters
    ----------
    machine_name : str : 'pfe', 'stampede', 'bw', 'peloton'
    directory_from : str : directory to copy from
    directory_to : str : local directory to put snapshots
    snapshot_indices : int or list : index[s] of snapshots to transfer
    '''
    snapshot_name_base = 'snap*_{:03d}*'

    directory_from = ut.io.get_path(simulation_directory_from) + 'output/'
    directory_to = ut.io.get_path(simulation_directory_to) + 'output/.'

    if np.isscalar(snapshot_indices):
        snapshot_indices = [snapshot_indices]

    snapshot_path_names = ''
    for snapshot_index in snapshot_indices:
        snapshot_path_names += (
            directory_from + snapshot_name_base.format(snapshot_index) + ' ')

    command = 'rsync -ahvP --size-only '
    command += '{}:"{}" {}'.format(machine_name, snapshot_path_names, directory_to)
    print('\n* executing:\n{}\n'.format(command))
    os.system(command)


def rsync_simulation_files(
    machine_name, directory_from='/oldscratch/projects/xsede/GalaxiesOnFIRE', directory_to='.'):
    '''
    Use rsync to copy simulation files.

    Parameters
    ----------
    machine_name : str : 'pfe', 'stampede', 'bw', 'peloton'
    directory_from : str : directory to copy from
    directory_to : str : directory to put files
    '''
    excludes = [
        'output/',
        'restartfiles/',

        'ewald_spc_table_64_dbl.dat',
        'spcool_tables/',
        'TREECOOL',

        'energy.txt',
        'balance.txt',
        'GasReturn.txt',
        'HIIheating.txt',
        'MomWinds.txt',
        'SNeIIheating.txt',

        '*.ics',

        'snapshot_scale-factors.txt',
        'submit_gizmo*.py',

        '*.bin',
        '*.particles',

        '*.bak',
        '*.err',
        '*.pyc',
        '*.o',
        '*.pro',
        '*.perl',
        '.ipynb_checkpoints',
        '.slurm',
        '.DS_Store',
        '*~',
        '._*',
        '#*#',
    ]

    directory_from = machine_name + ':' + ut.io.get_path(directory_from)
    directory_to = ut.io.get_path(directory_to)

    command = 'rsync -ahvP --size-only '

    arguments = ''
    for exclude in excludes:
        arguments += '--exclude="{}" '.format(exclude)

    command += arguments + directory_from + ' ' + directory_to + '.'
    print('\n* executing:\n{}\n'.format(command))
    os.system(command)


#===================================================================================================
# delete files
#===================================================================================================
def delete_snapshots(
    snapshot_directory='output', snapshot_index_limits=[1, 599], delete_halos=False):
    '''
    Delete all snapshots in given directory within snapshot_index_limits,
    except for those in snapshot_indices_keep list.

    Parameters
    ----------
    snapshot_directory : str : directory of snapshots
    snapshot_index_limits : list : min and max snapshot indices to delete
    delete_halos : bool : whether to delete halo catalog files at same snapshot times
    '''
    snapshot_name_base = 'snap*_{:03d}*'
    if not snapshot_directory:
        snapshot_directory = 'output/'

    halo_name_base = 'halos_{:03d}*'
    halo_directory = 'halo/rockstar_dm/catalog/'

    if snapshot_directory[-1] != '/':
        snapshot_directory += '/'

    if snapshot_index_limits is None or not len(snapshot_index_limits):
        snapshot_index_limits = [1, 599]
    snapshot_indices = np.arange(snapshot_index_limits[0], snapshot_index_limits[1] + 1)

    print()
    for snapshot_index in snapshot_indices:
        if snapshot_index not in snapshot_indices_keep:
            snapshot_name = snapshot_directory + snapshot_name_base.format(snapshot_index)
            print('* deleting:  {}'.format(snapshot_name))
            os.system('rm -rf {}'.format(snapshot_name))

            if delete_halos:
                halo_name = halo_directory + halo_name_base.format(snapshot_index)
                print('* deleting:  {}'.format(halo_name))
                os.system('rm -rf {}'.format(halo_name))
    print()


#===================================================================================================
# running from command line
#===================================================================================================
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise OSError('specify function to run: compress, globus, rsync, delete')

    function_kind = str(sys.argv[1])

    assert ('compress' in function_kind or 'rsync' in function_kind or 'globus' in function_kind or
            'delete' in function_kind)

    if 'compress' in function_kind:
        directory = 'output'
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])

        snapshot_index_max = 600
        if len(sys.argv) > 3:
            snapshot_index_max = int(sys.argv[3])
            snapshot_index_limits = [0, snapshot_index_max]

        Compress.compress_snapshots(directory, snapshot_index_limits=snapshot_index_limits)

    elif 'globus' in function_kind:
        directory = '.'
        if len(sys.argv) > 2:
            directory = str(sys.argv[2])
        Globus.submit_transfer(directory)

    elif 'rsync' in function_kind:
        if len(sys.argv) < 5:
            raise OSError(
                'imports: machine_name simulation_directory_from simulation_directory_to')

        machine_name = str(sys.argv[2])
        simulation_directory_from = str(sys.argv[3])
        simulation_directory_to = str(sys.argv[4])

        rsync_simulation_files(machine_name, simulation_directory_from, simulation_directory_to)
        rsync_snapshots(machine_name, simulation_directory_from, simulation_directory_to)

    elif 'delete' in function_kind:
        directory = 'output'
        if len(sys.argv) > 3:
            directory = str(sys.argv[3])

        snapshot_index_limits = None
        if len(sys.argv) > 4:
            snapshot_index_limits = [int(sys.argv[4]), int(sys.argv[5])]

        delete_snapshots(directory, snapshot_index_limits)
