import json

import requests

HUE_NUPNP_URL = "https://www.meethue.com/api/nupnp"

class APIException(Exception):
    pass

class HueAPI(object):

    def __init__(self, username):
        self.username = username

        self.ip = self.discover_hub_ip()

    @property
    def base_url(self):
        return "http://{}/api/{}".format(self.ip, self.username)

    def get_groups(self):
        url = "{}/groups".format(self.base_url)
        try:
            r = requests.get(url)
        except:
            raise APIException("Failed to send group get GET")

        try:
            return list(r.json().keys())
        except:
            raise APIException("Failed to decode group get json response")

    def set_group(self, group_id, state):
        url = "{}/groups/{}/action".format(self.base_url, group_id)
        try:
            r = requests.put(url, data=json.dumps({"on": state}))
        except:
            raise APIException("Failed to send group set PUT")

    def set_groups(self, state):
        for group in self.get_groups():
            self.set_group(group, state)

    def discover_hub_ip(self):
        try:
            r = requests.get(HUE_NUPNP_URL)
        except:
            raise APIException("Failed to send hub ip GET")

        try:
            json_resp = r.json()
        except:
            raise APIException("Failed to decode hub ip json response")

        if len(json_resp) > 0:
            return [0]['internalipaddress']
        else:
            raise APIException("Failed to find hub ip")


def _main():
    pass


if __name__ == '__main__':
    _main()
