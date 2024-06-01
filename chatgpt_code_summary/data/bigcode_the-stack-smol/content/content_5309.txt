import os
import requests

WEBHOOK_URL = os.environ['DJANGO_WEBHOOK_URL']


def send_message(author_name: str, message: str) -> bool:
    json_data = {
        # 'content': f'**{name}**\n\n{message}'
        'embeds': [
            {
                'author': {
                    'name': author_name,
                },
                'title': 'New message',
                'description': message
            }
        ]
    }

    response = requests.post(WEBHOOK_URL, json=json_data)
    return 200 <= response.status_code < 300
