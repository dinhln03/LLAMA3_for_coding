import glob
import shutil
import subprocess
import os
import sys
import argparse


# Read and save metadata from file
def exiftool_metadata(path):
    metadata = {}
    exifToolPath = 'exifTool.exe'
    ''' use Exif tool to get the metadata '''
    process = subprocess.Popen(
        [
            exifToolPath,
            path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    ''' get the tags in dict '''
    for tag in process.stdout:
        tag = tag.strip()
        key = tag[:tag.find(':')].strip()
        value = tag[tag.find(':') + 1:].strip()
        metadata[key] = value
    return metadata


class File:
    def __init__(self, path):
        self.metadata = exiftool_metadata(path)

    def _get_file_metadata(self, key, no=''):
        if key in self.metadata:
            return self.metadata[key]
        else:
            return no

    def copyCore(self, source, dst_dir: str, copy_duplicate=False):

        logs = []
        # if value of metadata not exists - folder name
        no_metadata = 'none'
        date = File._get_file_metadata(self, 'Date/Time Original')
        if date == '':
            date = File._get_file_metadata(self, 'Create Date', no_metadata)

        mime_type = File._get_file_metadata(self, 'MIME Type', no_metadata)
        dst_dir += f'''/{mime_type[:mime_type.find('/')]}/{date[:4]}/{date[5:7]}'''

        filename = File._get_file_metadata(self, 'File Name')
        f_name = filename
        dst = dst_dir + '/' + filename

        # File with the same name exists in dst. If source and dst have same size then determines 'copy_exists'
        if os.path.isfile(dst):
            i = 0
            f_pth = File(dst)
            if_same_size: bool = f_pth._get_file_metadata("File Size") == File._get_file_metadata(self, 'File Size')
            if (not if_same_size) or copy_duplicate:
                while os.path.isfile(dst):
                    filename = f'''{f_name[:f_name.find('.')]}_D{str(i)}.{File._get_file_metadata(self, 'File Type Extension')}'''
                    dst = f'''{dst_dir}/{filename}'''
                    i = i + 1
                if if_same_size:
                    logs.append(f"Warning: file already exists but I must copy all files"
                                f" [copy_duplicate={copy_duplicate}], so I try do it ...")
                else:
                    logs.append(f"Warning: file already exists but have other size, so I try copy it ...")

            else:
                logs.append(f"Warning: file already duplicate [copy_exists={copy_duplicate}]." 
                            f"\nCopy aboard: {source} -> {dst}")
                return logs

        try:
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)
                logs.append(f"New directory created: {dst_dir}")
            shutil.copy(source, dst)
            logs.append(f'''Copy done: {source} -> {dst}''')
        except Exception as e:
            logs.append(f'''Copy error [{e}]: {source} ->  {dst}''')

        return logs


def main():
    # Arguments from console
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help="Obligatory: source directory path")
    parser.add_argument('-d', help="Obligatory: destination folder path")
    parser.add_argument('-e', help="Obligatory: copy duplicate files (T/True/F/False)")
    args = parser.parse_args(sys.argv[1:])

    # Setup variable
    source_dir = args.s
    dst_dir = args.d
    df = {
        "T": True,
        "TRUE": True,
        "F": False,
        "FALSE": False
    }
    try:
        copy_duplicate = df.get(args.e.upper(), False)
    except AttributeError:
        copy_duplicate = False
        print(f"app.py: error: unrecognized arguments. Use -h or --help to see options")
        exit(1)

    # Number of log
    l_lpm = 0

    # source_dir = 'C:/Users'
    # dst_dir = 'C:/Users'
    # copy_duplicate = False

    for f_inx, source in enumerate(glob.glob(source_dir + '/**/*.*', recursive=True)):
        try:
            f = File(source)
            print("----------")
            for log in f.copyCore(source, dst_dir, copy_duplicate):
                l_lpm = l_lpm + 1
                print(f'''{str(l_lpm)}.{f_inx + 1}) {log}''')
        except Exception as e:
            print(f'Copy error [{e}]: {source}')


if __name__ == '__main__':
    main()
