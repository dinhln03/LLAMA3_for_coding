import geopy

# Here we just tested geopy functionality.

place = "kuusalu"
locator = geopy.Nominatim(user_agent="myGeocoder")
location = locator.geocode(place)
print(place + ":")
print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
