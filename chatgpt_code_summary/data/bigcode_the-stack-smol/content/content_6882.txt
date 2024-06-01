import re

from videos_id.platform import Platform

class Vimeo(Platform):
    def __init__(self):
        self.platform = "Vimeo"

    def check_url(self, url):
        pattern = r'https?:\/\/(?:www\.|player\.)?vimeo.com\/(?:channels\/(?:\w+\/)?|groups\/(?:[^\/]*)\/videos\/|album\/(?:\d+)\/video\/|video\/|)(\d+)(?:$|\/|\?)'

        match = re.search(pattern, url, re.IGNORECASE)

        if match:
            return match.group(1)
        else:
            return None