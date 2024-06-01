from ttv_api import *


@dataclass
class Channel:
    broadcaster_id: str
    broadcaster_login: str
    broadcaster_name: str
    game_name: str
    game_id: str
    broadcaster_language: str
    title: str
    delay: int


def get_channels(*channel_ids: str) -> Optional[list[Channel]]:
    params = "?"
    for channel_id in channel_ids:
        params += f"broadcaster_id={channel_id}&"

    http = urllib3.PoolManager()
    r = http.request(
        "GET",
        URL.channels.value + params,
        headers=HEADER,
    )
    if r.status != 200:
        return None

    data = json.loads(r.data.decode("utf-8"))["data"]
    channels: list[Channel] = []

    for channel in data:
        channels.append(
            Channel(
                channel["broadcaster_id"],
                channel["broadcaster_login"],
                channel["broadcaster_name"],
                channel["game_name"],
                channel["game_id"],
                channel["broadcaster_language"],
                channel["title"],
                channel["delay"],
            )
        )
    return channels
