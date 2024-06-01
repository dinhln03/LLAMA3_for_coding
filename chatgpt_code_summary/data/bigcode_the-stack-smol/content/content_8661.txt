from aiogram.dispatcher.filters.state import StatesGroup, State


class Support(StatesGroup):
    add_text = State()
    reply_msg = State()


class AdminPanel(StatesGroup):
    text = State()


class SendMsg(StatesGroup):
    id = State()
    msg = State()


class ChangeText(StatesGroup):
    text = State()
