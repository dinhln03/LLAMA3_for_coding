from datetime import timedelta
import json
from os import listdir
from os.path import isfile, join
import pr0gramm
import logging

__author__ = "Peter Wolf"
__mail__ = "pwolf2310@gmail.com"
__date__ = "2016-12-26"

LOG = logging.getLogger(__name__)


class DataSources:
    IMAGE, THUMBNAIL, FULL_SIZE = range(3)


class DataCollector:

    """ The DataCollector retrieves relevant data from
        pr0gramm and saves it locally.
    """

    def __init__(self, api, last_id=None):
        self.api = api
        self.last_id = last_id
        self.age_threshold = timedelta(hours=5)
        self.min_num_of_tags = 5
        self.search_forwards = True
        self.media_directory = "/tmp"
        self.data_source = DataSources.IMAGE
        self.annotation_file = "/tmp/annotation.txt"
        self.json_dir = "/tmp"
        self.download_media = True
        self.save_json = False
        self.use_local_storage = False
        self.last_batch_size = None

    def setAgeThreshold(self, days=0, hours=5, minutes=0, seconds=0):
        self.age_threshold = timedelta(
            days=days, hours=hours, minutes=minutes, seconds=seconds)

    def setMinimumNumberOfTags(self, threshold):
        self.min_num_of_tags = threshold

    def setLastId(self, last_id):
        self.last_id = last_id

    def getLastId(self):
        return self.last_id

    def useBackwardsSearch(self):
        self.search_forwards = False

    def useForwardsSearch(self):
        self.search_forwards = True

    def setMediaDirectory(self, directory):
        self.media_directory = directory

    def setDataSource(self, source):
        self.data_source = source

    def setAnnotationFile(self, annotation_file):
        self.annotation_file = annotation_file

    def setJsonDir(self, directory):
        self.json_dir = directory

    def setDownloadMedia(self, download_media):
        self.download_media = download_media

    def setSaveJSON(self, save_json):
        self.save_json = save_json

    def setUseLocalStorage(self, use_local_storage):
        self.use_local_storage = use_local_storage

    def getSizeOfLastBatch(self):
        return self.last_batch_size

    def download(self, item):
        if self.data_source == DataSources.IMAGE:
            return self.api.downloadMedia(
                item, save_dir=self.media_directory, file_name=item.id)
        elif self.data_source == DataSources.THUMBNAIL:
            return self.api.downloadThumbnail(
                item, save_dir=self.media_directory, file_name=item.id)
        elif self.data_source == DataSources.FULL_SIZE:
            return self.api.downloadFullsize(
                item, save_dir=self.media_directory, file_name=item.id)
        else:
            print "No valid data source chosen:", str(self.data_source)
            return None

    def writeAnnotation(self, item, media_path):
        # Read the current annotation file
        content = []
        if isfile(self.annotation_file):
            with open(self.annotation_file, "r") as f:
                content = f.readlines()

        # write every item as a line with the following structure:
        # ID;IMAGE_PATH;AMOUNT_OF_TAGS;...TAG_TEXT;TAG_CONFIDENCE;...
        new_line = str(item.id) + ";"
        new_line += str(media_path) + ";"
        new_line += str(len(item.tags)) + ";"
        new_line += ";".join([str(tag.getText()) + ";" +
                              str(tag.getConfidence()) for tag in item.tags])

        # Check if the item already has an entry in the annotation file
        # and replace it.
        contained = False
        for i in range(len(content)):
            if content[i].strip().startswith(str(item.id)):
                content[i] = new_line
                contained = True
                break

        # If no entry already exists, add a new line for the item
        if not contained:
            content.append(new_line)

        # Write the new content to the file.
        with open(self.annotation_file, "w") as f:
            for line in content:
                f.write(line.strip() + "\n")

    def getItemsFromAPI(self):
        if self.search_forwards:
            return self.api.getItemsNewer(self.last_id)
        else:
            return self.api.getItemsOlder(self.last_id)

    def getItemsFromLocalStorage(self):
        json_files = [join(self.json_dir, f) for f in listdir(self.json_dir)
                      if isfile(join(self.json_dir, f)) and f.endswith(".json")]

        data = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                json_item = json.load(f)
                item = pr0gramm.Item.Item.parseFromJSON(json_item)
                if not self.last_id \
                        or (self.search_forwards and item.getSortId() > self.last_id) \
                        or (not self.search_forwards and item.getSortId() < self.last_id):
                    data.append(item)
        data.sort(reverse=True)
        return data

    def collectDataBatch(self, data=[]):
        # retrieve data if none has been given
        if not data:
            if self.use_local_storage:
                data = self.getItemsFromLocalStorage()
            else:
                data = self.getItemsFromAPI()

        if not data:
            return

        # filter data based on age and tags
        valid_data = []
        for item in data:
            if item.getAge() >= self.age_threshold and len(item.tags) > 0:
                valid_data.append(item)

        # save size of collected data batch
        self.last_batch_size = len(valid_data)
        if not valid_data:
            return

        # save id of last item to fit age criteria in search direction
        if self.search_forwards:
            self.last_id = valid_data[0].getSortId()
        else:
            self.last_id = valid_data[-1].getSortId()

        for item in valid_data:
            if self.download:
                # download media
                target_path = self.download(item)
                if target_path:
                    # write id(s), link to media and tags to file
                    self.writeAnnotation(item, target_path)
            if self.save_json:
                with open(self.json_dir + "/" + str(item.id) + ".json", "w") as f:
                    json.dump(item.asDict(), f)

        return self.last_id
