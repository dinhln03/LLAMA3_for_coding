import os
from telethon import TelegramClient, events, sync
from dotenv import load_dotenv


sleeping_txt = "I'm sleeping right now, get back to you when I wake up"
turned_on = False
already_messaged_users = set()

if __name__ == "__main__":
    load_dotenv(override=True, verbose=True)
    api_id = int(os.getenv("API_ID"))
    api_hash = os.getenv("API_HASH")
    session_name = os.getenv("SESSION_NAME")


    with TelegramClient(session_name, api_id, api_hash) as client:
        client.send_message('me' , 'BRB online!')

        # Turns bot on and off
        async def toggle_brb(event, turn_on):
            global turned_on
            turned_on = turn_on
            sender = await event.message.get_sender()
            await event.reply(f"BRB on: {turned_on}")

        # /sleep : turns bot on
        @client.on(events.NewMessage(pattern='(?i)\/sleep$'))
        async def enable(event):
            await toggle_brb(event, True)

        # /up : turns bot off
        @client.on(events.NewMessage(pattern='(?i)\/up$'))
        async def disable(event):
            global already_messaged_users
            already_messaged_users = set() # clear cache of messaged users
            await toggle_brb(event, False)

        @client.on(events.NewMessage(incoming=True))
        async def say_brb(event):
            global turned_on, already_messaged_users
            sender = await event.get_sender()

            if not event.is_private or not turned_on or \
                    sender.id in already_messaged_users:
                return

            already_messaged_users.add(sender.id)
            await event.reply(sleeping_txt)

        client.run_until_disconnected()
