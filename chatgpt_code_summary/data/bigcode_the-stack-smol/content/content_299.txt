import requests
import json
from datetime import datetime, timezone
from . utils import _extract_videos_necessary_details, _save_video_detils_in_db
from .models import ApiKeys
from . import config


def _get_api_key(): #getting different key w.r.t last used every time cron job starts.(load balanced)
    new_key = ApiKeys.objects.all().order_by('last_used').first()
    _reponse = ApiKeys.objects.filter(
        api_key=new_key.api_key).update(last_used=datetime.now(timezone.utc))
    return new_key.api_key


def get_recent_youtube_videos_details():
    params = {**config.params}
    params.update({'key': _get_api_key()})
    print('Prameters: ', params)
    youtube_api_response = requests.get(
        config.YOUTUBE_SEARCH_URL, params=params)
    print('Youtube API Response: ', youtube_api_response.text)
    youtube_api_response = json.loads(youtube_api_response.text)
    videos_details = _extract_videos_necessary_details(
        youtube_api_response.get('items', []))
    if videos_details:
        _response = _save_video_detils_in_db(videos_details)
    return videos_details
