#!/usr/bin/python
"""Code for backing up pictures and videos captured to Dropbox"""
import sys
import os
import glob
from os import listdir
from os.path import isfile, join
import subprocess
from Adafruit_IO import Client, RequestError
import base64

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def main(args):
    """Main function"""
    if args:
        folder_path = args[0]
    else:
        folder_path = '/var/lib/motion'
    if not os.path.exists(folder_path):
        print "Folder Path: " + folder_path + " doesn't exist, exiting."
        raise ValueError("Incorrect Parameters")

    #Get the camera name. Default to visible
    with open(os.path.join(__location__, 'camera_name.cfg'), 'r') as f:
        print("open")
        camera_name = f.read().strip()

    print("Camera name: " + camera_name)

    aio = Client('ryhajlo', 'b5fe0936d9a84629a2d49cd45858fc67')
    
    # Start handling pictures
    videos = get_videos(folder_path)
    pictures = get_pictures(folder_path)
    if pictures:
        # Upload the files to dropbox
        # Build our command to upload files
        command = []
        command.append('/home/pi/Dropbox-Uploader/dropbox_uploader.sh')
        command.append('upload')
        for picture in pictures:
            print "Will upload: " + picture
            command.append(picture)
        command.append('/camera/pictures/' + camera_name + '/')
        subprocess.call(command)
        print "Finished uploading pictures"

        # Do the same for videos
        command = []
        command.append('/home/pi/Dropbox-Uploader/dropbox_uploader.sh')
        command.append('upload')
        for video in videos:
            print "Will upload: " + video
            command.append(video)
        command.append('/camera/videos/' + camera_name + '/')
        subprocess.call(command)
        print "Finished uploading videos"

        command = []
        command.append("mogrify")
        command.append("-resize")
        command.append("320x240")
        command.append("/var/lib/motion/*.jpg")
        subprocess.call(command)

        latest_picture = max(pictures, key=os.path.getctime)
        print "The latest picture is: " + latest_picture
        with open(latest_picture, "rb") as imageFile:
            image_str = base64.b64encode(imageFile.read())
  
        print "Uploading latest to Adafruit IO"
        feed_name = 'pic-' + camera_name
        print("Feed Name: " + feed_name)
        aio.send(feed_name, image_str )
        print "Finished uploading to Adafruit IO"
    else:
        latest_picture = None
        print "No pictures"
    

    # Now that everything is uploaded, delete it all
    for picture in pictures:
        print "Deleting " + picture
        os.remove(picture)
    for video in videos:
        print "Deleting " + video
        os.remove(video)
    
def get_videos(folder_path):
    videos = glob.glob(join(folder_path, '*.avi'))
    return videos

def get_pictures(folder_path):
    print "Grabbing files from " + folder_path

    # Get the list of files in that directory
    pictures = glob.glob(join(folder_path, '2*.jpg')) # Use the leading 2 to prevent us from getting 'latest.jpg'
    latest_picture = max(pictures, key=os.path.getctime)

    return pictures

if __name__ == "__main__":
    main(sys.argv[1:])

