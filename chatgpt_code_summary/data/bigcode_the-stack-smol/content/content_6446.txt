from map.models import *
import requests

# initialize geo_info table, used to show choropleth map
def run():
    try:
        response = requests.get('http://www.ourd3js.com/map/china_provinces/beijing.json')
        json_result = response.json()

        for area in json_result.get('features'):
            properties = area.get('properties')
            id = properties.get('id')
            geometry = area.get('geometry')
            district = properties.get('name')
            Geoinfo.create(id, district, properties, geometry).save()
    except:
        print("Load failed!")
