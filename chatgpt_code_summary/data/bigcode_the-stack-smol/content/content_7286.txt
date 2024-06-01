
# this is a library that can be used to update and create parts
# of the bulkdata
import urllib
import json
import time

from django.conf import settings

from apps.bulk.models import Character, Corporation, Alliance
from apps.static.models import Crpnpccorporations
from connection import connection


#Make sure not to many requests are made to evewho.com
def who_connect():
    timestamp = int(time.time())
    if who_connect.timestamp + 30 >= timestamp:
        who_connect.counter += 1
        if who_connect.counter == settings.EVE_WHO_REQUESTS:
            time.sleep(30)
            who_connect.counter = 0
            who_connect.timestamp = timestamp
    else:
        who_connect.timestamp = timestamp
        who_connect.counter = 0
who_connect.timestamp = int(time.time())
who_connect.counter = 0


#get the basepart of the api url
def get_url(category, pk, page=0):
    return "http://evewho.com/api.php?type=%s&id=%d&page=%d" % (
        category,
        pk,
        page
    )


# get the data from url
def json_object(url):
    response = urllib.urlopen(url)
    data = json.loads(response.read())
    return data


#temp function
def remaining_alliances():
    id_list = []
    for alli in Alliance.objects.all():
        if not Corporation.objects.filter(allianceid=alli.allianceid).exists():
            id_list.append(alli.allianceid)

    for pk in id_list:
        pages = True
        page = 0
        while pages:
            who_connect()
            data = json_object(get_url("allilist", pk, page=page))
            for char in data['characters']:
                if not Character.objects.filter(
                    characterid=char['character_id']
                ).exists():
                    Character.objects.create(
                        characterid=char["character_id"],
                        corporationid=char["corporation_id"],
                        allianceid=char["alliance_id"],
                        name=char["name"],
                    )

                if not Corporation.objects.filter(
                    corporationid=char["corporation_id"]
                ).exists():
                    corp = getattr(connection, "corporationsheet")(
                        char["corporation_id"]
                    )

                    try:
                        corp = Corporation(
                            corporationid=corp.corporationID,
                            corporationname=corp.corporationName,
                            ticker=corp.ticker,
                            ceoid=corp.ceoID,
                            ceoname=corp.ceoName,
                            allianceid=corp.allianceID,
                            alliancename=corp.allianceName,
                            stationid=corp.stationID,
                            description=unicode(corp.description),
                            url=corp.url,
                            taxrate=int(corp.taxRate),
                            membercount=corp.memberCount,
                        )
                        corp.save()
                        print corp.corporationname
                    except Exception, e:
                        print e

            if len(data['characters']) == 200:
                page += 1
            else:
                pages = False
