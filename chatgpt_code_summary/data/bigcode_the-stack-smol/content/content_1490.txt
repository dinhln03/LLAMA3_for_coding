import requests
import os

class IntegrationDiscordDriver:

    _scope = ''
    _state = ''

    def scopes(self, scopes):
        pass

    def send(self, request,  state='', scopes=('identify',)):
        self._scope = scopes
        self._state = state
        return request.redirect('https://discordapp.com/api/oauth2/authorize?response_type=code&client_id={}&scope={}&state={}&redirect_uri={}'.format(
            os.getenv('DISCORD_CLIENT'),
            ' '.join(self._scope),
            self._state,
            os.getenv('DISCORD_REDIRECT'),
        ))

    def user(self, request):
        data = {
            'client_id': os.getenv('DISCORD_CLIENT'),
            'client_secret': os.getenv('DISCORD_SECRET'),
            'grant_type': 'authorization_code',
            'code': request.input('code'),
            'redirect_uri': os.getenv('DISCORD_REDIRECT')
        }

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        return requests.post('https://discordapp.com/api/oauth2/token', data, headers).json()

    def refresh(self, refresh_token):
        data = {
            'client_id': os.getenv('DISCORD_CLIENT'),
            'client_secret': os.getenv('DISCORD_SECRET'),
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
            'redirect_uri': os.getenv('DISCORD_REDIRECT')
        }

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        return requests.post('https://discordapp.com/api/oauth2/token', data, headers).json()
