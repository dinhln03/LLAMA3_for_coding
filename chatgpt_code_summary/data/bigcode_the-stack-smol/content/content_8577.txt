from workerInfra.enum import SchedulerTriggerEnum, EnvEnum
from workerInfra.domain import LoggerInterface
from workerInfra.models import SchedulerJobModel
from workerService.envReaderService import EnvReaderService
from workerService.scheduler import SchedulerService
from workerService.logger import BasicLoggerService
from workerService.outputs import DiscordOutputService
from workerService.sources import (
    WorkerService,
    RedditWorkerService,
    YoutubeWorkerService,
    TwitterWorkerService,
    TwitchWorkerService,
    PokemonGoWorkerService,
    FFXIVWorkerService
)
from workerService.sources.rssWorkerService import RssWorkerService


class ApiEventsService():
    _env: EnvReaderService
    _logger: LoggerInterface
    _scheduler: SchedulerService
    _isDebug: bool

    def __init__(self) -> None:
        self._env = EnvReaderService()
        self._logger = BasicLoggerService()
        self._scheduler = SchedulerService()
        self._isDebug = self._env.getValue(EnvEnum.ISDEBUG)
        pass

    def startup(self) -> None:
        self._scheduler.addJob(self.enableSourceReddit())
        self._scheduler.addJob(self.enableSourceYoutube())
        self._scheduler.addJob(self.enableSourceTwitter())
        self._scheduler.addJob(self.enableSourceTwitch())
        self._scheduler.addJob(self.enableSourcePokemonGo())
        self._scheduler.addJob(self.enableSourceFFXIV())
        self._scheduler.addJob(self.enableSourceRss())

        self._scheduler.addJob(self.enableOutputDiscord())

        self._scheduler.start()

    def enableSourceRss(self) -> SchedulerJobModel:
        s = SchedulerJobModel(functionName=WorkerService(RssWorkerService()).init, trigger=SchedulerTriggerEnum.INTERVAL, minutes=30)
        if self._isDebug is True:
            s.trigger = SchedulerTriggerEnum.NONE
        if self._env.getValue(EnvEnum.RSSENABLED) is False:
            s.enabled = False
        return s

    def enableSourceReddit(self) -> SchedulerJobModel:
        s = SchedulerJobModel(functionName=WorkerService(RedditWorkerService()).init, trigger=SchedulerTriggerEnum.INTERVAL, minutes=25)
        if self._isDebug is True:
            s.trigger = SchedulerTriggerEnum.NONE
        if self._env.getValue(EnvEnum.REDDITENABLED) is False:
            s.enabled = False
        return s

    def enableSourceYoutube(self) -> SchedulerJobModel:
        s = SchedulerJobModel(functionName=WorkerService(YoutubeWorkerService()).init, trigger=SchedulerTriggerEnum.INTERVAL, minutes=35)
        if self._isDebug is True:
            s.trigger = SchedulerTriggerEnum.NONE
        if self._env.getValue(EnvEnum.YOUTUBEENABLED) is False:
            s.enabled = False
        return s

    def enableSourceTwitter(self) -> SchedulerJobModel:
        s = SchedulerJobModel(functionName=WorkerService(TwitterWorkerService()).init, trigger=SchedulerTriggerEnum.INTERVAL, minutes=37)
        if self._isDebug is True:
            s.trigger = SchedulerTriggerEnum.NONE
        if self._env.getValue(EnvEnum.TWITCHEANBLED) is False:
            s.enabled = False
        return s

    def enableSourceTwitch(self) -> SchedulerJobModel:
        s = SchedulerJobModel(functionName=WorkerService(TwitchWorkerService()).init, trigger=SchedulerTriggerEnum.INTERVAL, minutes=30)
        if self._isDebug is True:
            s.trigger = SchedulerTriggerEnum.NONE
        if self._env.getValue(EnvEnum.TWITCHEANBLED) is False:
            s.enabled = False
        return s

    def enableSourcePokemonGo(self) -> SchedulerJobModel:
        s = SchedulerJobModel(functionName=WorkerService(PokemonGoWorkerService()).init, trigger=SchedulerTriggerEnum.INTERVAL, minutes=35)
        if self._isDebug is True:
            s.trigger = SchedulerTriggerEnum.NONE
        if self._env.getValue(EnvEnum.POKEMONGOENABLED) is False:
            s.enabled = False
        return s

    def enableSourceFFXIV(self) -> SchedulerJobModel:
        s = SchedulerJobModel(functionName=WorkerService(FFXIVWorkerService()).init, trigger=SchedulerTriggerEnum.INTERVAL, minutes=45)
        if self._isDebug is True:
            s.trigger = SchedulerTriggerEnum.NONE
        if self._env.getValue(EnvEnum.FFXIVENABLED) is False:
            s.enabled = False
        return s

    def enableOutputDiscord(self) -> SchedulerJobModel:
        s = SchedulerJobModel(functionName=DiscordOutputService().init, trigger=SchedulerTriggerEnum.INTERVAL, minutes=3)
        if self._isDebug is True:
            s.minutes = 1
        return s

    def shutdown(self) -> None:
        pass
