from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *


app = Flask(__name__)
# LINE BOT info
line_bot_api = LineBotApi('QCynFfsDk7My1YN72sVQyvk6ArYkD2TUQW/pUxUQqllnGFNcqjZ8tKC+qMcVa2u4Lg1WmdUVLcS124tweaXtcVWLmK/thFH1NFUZL/Olev6ugLeKG4VUVd0ee8VUUgnrqqCZD+ZBpD6j61TRW2eJEgdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('f3b5d7b57ef1f4d1277aecd7f045db3d')

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    print(body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# Message event
@handler.add(MessageEvent)
def handle_message(event):
    message_type = event.message.type
    user_id = event.source.user_id
    reply_token = event.reply_token
    message = event.message.text
    line_bot_api.reply_message(reply_token, TextSendMessage(text = message))

import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)