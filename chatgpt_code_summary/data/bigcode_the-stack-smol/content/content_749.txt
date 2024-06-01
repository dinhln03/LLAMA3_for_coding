from asgiref.sync import sync_to_async
from channels.layers import get_channel_layer
from ....models import Participant
import humps

channel_layer = get_channel_layer()

def get_participant(room_channel_name, channel_name):
    participant = Participant.objects.get(
        channel_room__channel_name=room_channel_name,
        channel_name=channel_name
    )
    return participant

def get_participant_id(participant):
    return participant.id


async def broadcast_avatar_position(room_channel_name, channel_name, json_data):
    """
    Sends the new avatar's position to the users of the room.
    """

    type = json_data['type']
    payload = json_data['payload']
    position = payload["position"]
    animate = payload["animate"]

    # receive the participant that sent this message
    participant = await sync_to_async(get_participant)(room_channel_name, channel_name)
    participant_id = await sync_to_async(get_participant_id)(participant)

    # if this was for an avatar, then set participant's position to the payload data
    def set_participant_position():
        participant.x = position["x"]
        participant.y = position["y"]
        participant.direction_x = position["directionX"]
        participant.save()
    await sync_to_async(set_participant_position)()

    await channel_layer.group_send(
        room_channel_name,
        {
            'type': type,
            'payload': {
                "participant_id": participant_id,
                "position": position,
                "animate": animate,
            }
        }
    )

async def broadcast_avatar_state(room_channel_name, channel_name, json_data):
    """
    Sends the new avatar's state to the users of the room.
    """

    type = json_data['type']
    payload = json_data['payload']
    state = payload['value']

    # receive the participant that sent this message
    participant = await sync_to_async(get_participant)(room_channel_name, channel_name)
    participant_id = await sync_to_async(get_participant_id)(participant)
    
    await channel_layer.group_send(
        room_channel_name,
        {
            'type': humps.decamelize(type),
            'payload': {
                "participant_id": participant_id,
                "state": state
            }
        }
    )