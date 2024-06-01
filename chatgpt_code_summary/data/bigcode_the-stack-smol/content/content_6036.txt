#Written by Shitao Tang
# --------------------------------------------------------
import connectDB
import time,hashlib,logging

def sign_up(username,password):
    db=connectDB.database.getInstance()
    if len(username)<=20:
        return db.create_account(username,hashlib.sha224(password).hexdigest())
    else:
        return 'username must be less than 20 characters'
 
def account_authentication(username,password):
    db=connectDB.database.getInstance()
    result=db.authenticate_account(username,hashlib.sha224(password).hexdigest())
    if result:
        return hashlib.sha224(username+str(time.time())).hexdigest()
    elif result ==False:
        return None
    else:
        logging.error(result)
        
    
    
def check_keys(data,keys): #check whether a dictionary contains a list of keys
        for key in keys:
            if key not in data:
                return key

        return None

def check_float(value,min_value,max_value): #try to convert value to a float number and is between min_value and max_value
    try:
        value=float(value)
        if value>=min_value and value<=max_value:
            return value
        else:
            return None
    except ValueError:
        return None
       
def decode_xml(object_name,xml): #get the bounding box of the object in an image
    logging.info("begin to decode")
    bounding_box=[]
    #print xml
    import xml.etree.ElementTree as ET
    try:
        root=ET.fromstring(xml)
    except:
        return []

    for obj in root.findall('object'):
        if(obj.find('name').text==object_name):
            score=float(obj.find("score").text)
            bnd_box=obj.find('bndbox')
            xmin=int((bnd_box).find('xmin').text)
            ymin=int((bnd_box).find('ymin').text)
            xmax=int((bnd_box).find('xmax').text)
            ymax=int((bnd_box).find('ymax').text)
            bounding_box.append((xmin,ymin,xmax,ymax,score))
    return bounding_box

def coordinate_from_google_to_baidu(longitude,latitude): 
    return gcj02tobd09(longitude,latitude)

def coordinate_from_baidu_to_google(longitude,latitude):
    return bd09togcj02(longitude,latitude)

def check_connection_of_image_analysis_server(address):
    reponse=requests.get(address+"/ok")
    print address,reponse.text
    if reponse.text=="OK":
        return True
    else:
        return False

#the following code is copied from github
import json
import requests
import math

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  
a = 6378245.0  
ee = 0.00669342162296594323  


def geocode(address):
    geocoding = {'s': 'rsv3',
                 'key': key,
                 'city': 'china',
                 'address': address}
    res = requests.get(
        "http://restapi.amap.com/v3/geocode/geo", params=geocoding)
    if res.status_code == 200:
        json = res.json()
        status = json.get('status')
        count = json.get('count')
        if status == '1' and int(count) >= 1:
            geocodes = json.get('geocodes')[0]
            lng = float(geocodes.get('location').split(',')[0])
            lat = float(geocodes.get('location').split(',')[1])
            return [lng, lat]
        else:
            return None
    else:
        return None


def gcj02tobd09(lng, lat):
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * x_pi)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * x_pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return [bd_lng, bd_lat]


def bd09togcj02(bd_lon, bd_lat):
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gg_lng = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return [gg_lng, gg_lat]


def wgs84togcj02(lng, lat):
    """
    """
    if out_of_china(lng, lat): 
        return lng, lat
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


def gcj02towgs84(lng, lat):
    """
    """
    if out_of_china(lng, lat):
        return lng, lat
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
        0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
        0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)


