from ParseHandler import ParseHandler
from PathHandler import PathHandler
import paths

parser = ParseHandler()
pather = PathHandler()

# match subdirectories in both folders
pather.build_matching_subdir(paths.TOY_RAW, paths.TOY_CLEAN)

# get paths to folders in raw/ directory
dir_names = pather.get_dir_names(paths.TOY_RAW)
raw_dir_paths = pather.get_dir_paths(paths.TOY_RAW)
clean_dir_paths = pather.get_dir_paths(paths.TOY_CLEAN)


# iterate through the contents of each folder
for raw_dir_path, clean_dir_path in zip(raw_dir_paths, clean_dir_paths):
        
    # get raw file paths from each subdir
    file_names = pather.get_file_names(raw_dir_path)
    raw_file_paths = pather.get_file_paths(raw_dir_path)
    clean_file_paths = [clean_dir_path + file_name for file_name in file_names]
    
    # parse each raw_file into the clean_file
    for raw_file_path, clean_file_path in zip(raw_file_paths, clean_file_paths):
        parser.parse(raw_file_path, clean_file_path)

