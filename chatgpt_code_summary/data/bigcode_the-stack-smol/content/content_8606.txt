import os.path


class Settings:

    def __init__(self):
        self.entry_point = os.path.expanduser('facts.graft')
        self.userfacts = os.path.expanduser('~/.facts/user.yml')
        self.userpath = os.path.expanduser('~/.facts/grafts')

settings = Settings()
