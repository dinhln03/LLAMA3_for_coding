from mopidy import backend

import pykka

from mopidy_funkwhale import api, client, library, playback, playlists


class FunkwhaleBackend(pykka.ThreadingActor, backend.Backend):
    def __init__(self, config, audio):
        super(FunkwhaleBackend, self).__init__()
        self.api = api.FunkwhaleApi(config)
        self.client = client.FunkwhaleClient(self.api)
        self.audio = audio
        self.library = library.FunkwhaleLibraryProvider(backend=self)
        self.playback = playback.FunkwhalePlaybackProvider(audio=audio,
                                                           backend=self)
        self.playlists = playlists.FunkwhalePlaylistsProvider(backend=self)
        self.uri_schemes = ['funkwhale']

    def on_start(self):
        self.api.login()
