# -*- coding: utf-8 -*-
import os
import shutil


def make_empty_folder(folder_path:str):
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
        else:
            os.remove(folder_path)

    os.mkdir(folder_path)


def copy_files(from_path:str, to_path:str, extension:str):
    files = os.listdir(from_path)
    for file in files:
        if file.endswith(extension):
            shutil.copy(from_path + '/' + file, to_path)


def append_src(to_file, from_file: str):
    with open(from_file, 'r') as f:
        for line in f:
            to_file.write(line)
            

if __name__ == '__main__':
    project_name = 'viscomp_final'
    source_folder_name = 'JellyfishAquarium'
    src_folder = './' + source_folder_name
    out_folder = './' + project_name

    make_empty_folder(out_folder)

    for extension in ['png', 'jpg']:
        copy_files(src_folder, out_folder, extension)

    # combine all pde files into viscomp_final.pde
    with open(f'{out_folder}/{project_name}.pde', 'w') as f:
        append_src(f, f'{src_folder}/{source_folder_name}.pde')
        files = os.listdir(src_folder)
        for file in files:
            if file.endswith('.pde') and file != f'{source_folder_name}.pde':
                f.write('\n\n')
                append_src(f, src_folder + '/' + file)
