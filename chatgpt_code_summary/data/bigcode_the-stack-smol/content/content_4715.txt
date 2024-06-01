# https://github.com/theeko74/pdfc
# modified by brio50 on 2022/01/23, working with gs version 9.54.0

"""
Simple python wrapper script to use ghoscript function to compress PDF files.

Compression levels:
    0: default
    1: prepress
    2: printer
    3: ebook
    4: screen

Dependency: Ghostscript.
On MacOSX install via command line `brew install ghostscript`.
"""

import argparse
import subprocess
import os.path
import sys
import shutil


def compress(input_file_path, output_file_path, level=0, method=1):
    """Function to compress PDF via Ghostscript command line interface"""
    quality = {
        0: '/default',
        1: '/prepress',
        2: '/printer',
        3: '/ebook',
        4: '/screen'
    }

    # Check if valid path
    if not os.path.isfile(input_file_path):
        print(f"Error: invalid path for input file: {input_file_path}")
        sys.exit(1)

    # Check if file is a PDF by extension
    if input_file_path.split('.')[-1].lower() != 'pdf':  # not sure this is the most robust solution
        print(f"Error: input file is not a PDF: {input_file_path}")
        sys.exit(1)

    gs = get_ghostscript_path()
    file_name = input_file_path.split('/')[-1]  # everything after last '/'
    print("Compressing PDF \"{}\"...".format(file_name))

    if method == 1:
        # https://gist.github.com/lkraider/f0888da30bc352f9d167dfa4f4fc8213
        cmd = [gs, '-sDEVICE=pdfwrite',
               '-dNumRenderingThreads=2',
               '-dPDFSETTINGS={}'.format(quality[level]),
               '-dCompatibilityLevel=1.5',
               '-dNOPAUSE', '-dQUIET', '-dBATCH', '-dSAFER',
               # font settings
               '-dSubsetFonts=true',
               '-dCompressFonts=true',
               '-dEmbedAllFonts=true',
               # color format`
               '-sProcessColorModel=DeviceRGB',
               '-sColorConversionStrategy=RGB',
               '-sColorConversionStrategyForImages=RGB',
               '-dConvertCMYKImagesToRGB=true',
               # image resample
               '-dDetectDuplicateImages=true',
               '-dColorImageDownsampleType=/Bicubic',
               '-dColorImageResolution=300',
               '-dGrayImageDownsampleType=/Bicubic',
               '-dGrayImageResolution=300',
               '-dMonoImageDownsampleType=/Subsample',
               '-dMonoImageResolution=300',
               '-dDownsampleColorImages=true',
               # preset overrides
               '-dDoThumbnails=false',
               '-dCreateJobTicket=false',
               '-dPreserveEPSInfo=false',
               '-dPreserveOPIComments=false',
               '-dPreserveOverprintSettings=false',
               '-dUCRandBGInfo=/Remove',
               '-sOutputFile={}'.format(output_file_path),
               input_file_path]
    elif method == 2:
        cmd = [gs, '-sDEVICE=pdfwrite',
               '-dNumRenderingThreads=2',
               '-dPDFSETTINGS={}'.format(quality[level]),
               '-dCompatibilityLevel=1.4',
               '-dNOPAUSE', '-dQUIET', '-dBATCH', '-dSAFER',
               '-dDetectDuplicateImages=true',
               '-sOutputFile={}'.format(output_file_path),
               input_file_path]

    try:
        # execute
        subprocess.call(cmd, stderr=sys.stdout)
    except:
        # print ghostscript command for debug
        print(" ".join(cmd))

    if not os.path.exists(output_file_path):
        raise Exception(f"Ghostscript failed to create {output_file_path}, time to debug...\n",
                        " ".join(cmd))

    initial_size = round(os.path.getsize(input_file_path) / (1024 * 1024), 2)
    final_size = round(os.path.getsize(output_file_path) / (1024 * 1024), 2)
    ratio = round(100 - ((final_size / initial_size) * 100), 1)

    print(f"Initial file size is {initial_size}MB",
          f"; Final file size is {final_size}MB",
          f"; Compression Ratio = {ratio}%\n")

    if final_size > initial_size and method == 1:
        print('-' * 100)
        print('Compression Failed\nTrying another ghostscript compression method...')
        print('-' * 100)
        info = compress(input_file_path, output_file_path, 4, 2)
        initial_size = info[0]
        final_size = info[1]
        ratio = info[2]

    return [initial_size, final_size, ratio]


def get_ghostscript_path():
    gs_names = ['gs', 'gswin32', 'gswin64']
    for name in gs_names:
        if shutil.which(name):
            return shutil.which(name)
    raise FileNotFoundError(f'No GhostScript executable was found on path ({"/".join(gs_names)})')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input', help='Relative or absolute path of the input PDF file')
    parser.add_argument('-o', '--out', help='Relative or absolute path of the output PDF file')
    parser.add_argument('-c', '--compress', type=int, help='Compression level from 0 to 4')
    parser.add_argument('-b', '--backup', action='store_true', help="Backup the old PDF file")
    parser.add_argument('--open', action='store_true', default=False,
                        help='Open PDF after compression')
    args = parser.parse_args()

    # In case no compression level is specified, default is 2 '/ printer'
    if not args.compress:
        args.compress = 2
    # In case no output file is specified, store in temp file
    if not args.out:
        args.out = 'temp.pdf'

    # Run
    compress(args.input, args.out, power=args.compress)

    # In case no output file is specified, erase original file
    if args.out == 'temp.pdf':
        if args.backup:
            shutil.copyfile(args.input, args.input.replace(".pdf", "_BACKUP.pdf"))
        shutil.copyfile(args.out, args.input)
        os.remove(args.out)

    # In case we want to open the file after compression
    if args.open:
        if args.out == 'temp.pdf' and args.backup:
            subprocess.call(['open', args.input])
        else:
            subprocess.call(['open', args.out])


if __name__ == '__main__':
    main()
