#!/usr/bin/env python                                            
#
# mri_convert_ppc64 ds ChRIS plugin app
#
# (c) 2016-2019 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#


import os
import sys
sys.path.append(os.path.dirname(__file__))

# import the Chris app superclass
from chrisapp.base import ChrisApp


Gstr_title = """


                _                                _                         ____    ___ 
               (_)                              | |                       / ___|  /   |
 _ __ ___  _ __ _   ___ ___  _ ____   _____ _ __| |_     _ __  _ __   ___/ /___  / /| |
| '_ ` _ \| '__| | / __/ _ \| '_ \ \ / / _ \ '__| __|   | '_ \| '_ \ / __| ___ \/ /_| |
| | | | | | |  | || (_| (_) | | | \ V /  __/ |  | |_    | |_) | |_) | (__| \_/ |\___  |
|_| |_| |_|_|  |_| \___\___/|_| |_|\_/ \___|_|   \__|   | .__/| .__/ \___\_____/    |_/
               ______                             ______| |   | |                      
              |______|                           |______|_|   |_|                      
                                                                                       
                                                                                       
                                                                                       
                                                                                       

"""

Gstr_synopsis = """


    NAME

       mri_convert_ppc64.py 

    SYNOPSIS

        python mri_convert_ppc64.py                                     \\
            [-h] [--help]                                               \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v <level>] [--verbosity <level>]                          \\
            [--version]                                                 \\
            [--inputFile <inputFile>]                                   \\
            [--outputFile <outputFile>]                                 \\
            [--executable <executable>]                                 \\
            [--execArgs <execArgs>]                                     \\
            <inputDir>                                                  \\
            <outputDir> 

    BRIEF EXAMPLE

        * Bare bones execution

            mkdir in out && chmod 777 out
            python mri_convert_ppc64.py   \\
                                in    out

    DESCRIPTION

        `mri_convert_ppc64.py` calls an underlying executable 
        (typically 'mri_convert') and passes it an input and output spec.

    ARGS

        [--inputFile <inputFile>]                               
        The input file, relative to <inputDir>.

        [--outputFile <outputFile>]     
        The output file, relative to <outpufDir>.                               

        [--executable <executable>]             
        The actual executable to run.                   

        [--execArgs <execArgs>] 
        Additional executable-specific command line args.
                                
        [-h] [--help]
        If specified, show help message and exit.
        
        [--json]
        If specified, show json representation of app and exit.
        
        [--man]
        If specified, print (this) man page and exit.

        [--meta]
        If specified, print plugin meta data and exit.
        
        [--savejson <DIR>] 
        If specified, save json representation file to DIR and exit. 
        
        [-v <level>] [--verbosity <level>]
        Verbosity level for app. Not used currently.
        
        [--version]
        If specified, print version number and exit. 

"""


class Mri_convert_ppc64(ChrisApp):
    """
    This calls a pre-built PPC64 'mri_convert' that is housed in a base container..
    """
    AUTHORS                 = 'BU-2019-Power9 (dev@babyMRI.org)'
    SELFPATH                = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC                = os.path.basename(__file__)
    EXECSHELL               = 'python3'
    TITLE                   = 'A PowerPPC plugin to run the FreeSurfer mri_convert'
    CATEGORY                = ''
    TYPE                    = 'ds'
    DESCRIPTION             = 'This calls a pre-built PPC64 mri_convert that is housed in a base container.'
    DOCUMENTATION           = 'http://wiki'
    VERSION                 = '0.1'
    ICON                    = '' # url of an icon image
    LICENSE                 = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MAX_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT           = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT           = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """

        self.add_argument('--executable', 
                           dest         = 'executable',
                           type         = str, 
                           optional     = True,
                           help         = 'the conversion program to use',
                           default      = '/usr/bin/mri_convert')

        self.add_argument('--inputFile', 
                           dest         = 'inputFile', 
                           type         = str, 
                           optional     = True,
                           help         = 'the input file',
                           default      = '')

        self.add_argument('--outputFile', 
                           dest         = 'outputFile', 
                           type         = str, 
                           optional     = True,
                           help         = 'the output file',
                           default      = '')

        self.add_argument('--execArgs',
                           dest         = 'execArgs',
                           type         = str, 
                           optional     = True,
                           help         = 'additonal arguments for the chosen executable',
                           default      = '')

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        if not len(options.inputFile):
                print("ERROR: No input file has been specified!")
                print("You must specify an input file relative to the input directory.")
                sys.exit(1)

        if not len(options.outputFile):
                print("ERROR: No output file has been specified!")
                print("You must specicy an output file relative to the output directory.")
                sys.exit(1)

        str_cmd = '%s %s %s/%s %s/%s' % (       options.executable, 
                                                options.execArgs,
                                                options.inputdir, 
                                                options.inputFile, 
                                                options.outputdir, 
                                                options.outputFile)
        os.system(str_cmd)

    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_title)
        print(Gstr_synopsis)


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = Mri_convert_ppc64()
    chris_app.launch()
