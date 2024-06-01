import os
import requests

# configurations to be adjusted
# 1. put here URL (see textfile)
base_url = "https://data-dataref.ifremer.fr/stereo/AA_2015/2015-03-05_10-35-00_12Hz/input/cam1/"
# 2. decide which (range of) images
start = 0
end = 149
# 3. name folder to save images to, best take from url (change "/" to "_")
download_folder = 'AA_2015_2015-03-05_10-35-00_12Hz'
img_appendix = "_01" # as the datasat is providing stereo, we only need mono, not to be changed

#create a download folder if not yet existing
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, download_folder)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

# run through all url to download images individually, same file name as in original dataset
#start to uncomment
# while start <= end:
#     img_name = f'{start:06d}' + img_appendix + '.tif'
#     #print(f"image_name is: " + img_name)
#     url = base_url + img_name
#     r = requests.get(url, allow_redirects=True)
#     # print(f"loading url: " + url)
#
#     # save image in download_folder
#     path_dest = os.path.join(final_directory, img_name)
#     open(path_dest, 'wb').write(r.content)
#
#     start += 1
#
# print("Done")
#end to uncomment


#Alternative with .txt list (AA-Videos need it)

with open("/Users/rueskamp/Documents/Studium SE/05_WS21/Projekt_See/codebase/dataset_preparation/AA_2015_2015-03-05_10-35-00_12Hz.txt", "r") as f:
    list2 = []
    for item in f:
        one, two = item.split(">", 1)
        img_name = one
        #     #print(f"image_name is: " + img_name)
        url = base_url + img_name
        r = requests.get(url, allow_redirects=True)
        # print(f"loading url: " + url)
        #
        # save image in download_folder
        path_dest = os.path.join(final_directory, img_name)
        open(path_dest, 'wb').write(r.content)

print("Done")