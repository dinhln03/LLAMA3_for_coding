import os

class FileCredentials:

    def __init__(self, credentials_file):
        if credentials_file == None:
            credentials_file = os.path.expanduser("~") + "/.pingboard"

        self.credentials_file = credentials_file
        self.client_id = None
        self.client_secret = None

    def load(self):
        try:
            credentials = dict(line.strip().split('=') for line in open(self.credentials_file))
            self.client_id = credentials['client_id']
            self.client_secret = credentials['client_secret']
            return True
        except Exception as e:
            return False


class ArgsCredentials:

    def __init__(self, id_key, secret_key, **kwargs):
        self.client_id = None
        self.client_secret = None
        try:
            self.client_id = kwargs[id_key]
            self.client_secret = kwargs[secret_key]
        except KeyError:
            pass

    def load(self):
        return self.client_id != None and self.client_secret != None;

class Credentials:

    def __init__(self, **kwargs):

        self.chain = [
            ArgsCredentials('client_id', 'client_secret',
                **kwargs),
            ArgsCredentials('PINGBOARD_CLIENT_ID', 'PINGBOARD_CLIENT_SECRET',
                **os.environ),
            FileCredentials(kwargs.get('credentials_file'))
        ]

    def load(self):
        loaded_credentials = None

        for credentials in self.chain:
            if credentials.load():
                loaded_credentials = credentials
                break

        if not loaded_credentials:
            return False

        self.client_id = loaded_credentials.client_id
        self.client_secret = loaded_credentials.client_secret

        return True
