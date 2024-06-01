from audio import Stream, AudioSettings


class PhraseRecognizer(object):
    def __init__(self, config, audio_settings: AudioSettings):
        self._config = config
        self._audio_settings = audio_settings

    def get_config(self):
        return self._config

    def get_audio_settings(self) -> AudioSettings:
        return self._audio_settings

    async def recognize(self, stream: Stream, recv_callback):
        raise Exception('Not implemented "recognize"')


class HotwordRecognizer(object):
    def __init__(self, config):
        self._config = config

    def get_audio_settings(self) -> AudioSettings:
        raise Exception('Not implemented "get_audio_settings"')

    def start(self):
        pass

    def is_hotword(self, raw_frames) -> bool:
        raise Exception('Not implemented "is_hotword"')


class VADRecognizer(object):
    def __init__(self, config):
        self._config = config

    def get_audio_settings(self) -> AudioSettings:
        raise Exception('Not implemented "get_audio_settings"')

    def is_speech(self, raw_frames) -> bool:
        raise Exception('Not implemented "is_speech"')


class PhraseRecognizerConfig(object):
    def create_phrase_recognizer(self) -> PhraseRecognizer:
        raise Exception('Not implemented "create_phrase_recognizer"')


class HotwordRecognizerConfig(object):
    def create_hotword_recognizer(self) -> HotwordRecognizer:
        raise Exception('Not implemented "create_hotword_recognizer"')


class VADRecognizerConfig(object):
    def create_vad_recognizer(self) -> VADRecognizer:
        raise Exception('Not implemented "create_vad_recognizer"')
