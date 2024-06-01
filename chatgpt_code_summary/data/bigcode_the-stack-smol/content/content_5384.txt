"""Download handlers for different schemes"""

import logging

from twisted.internet import defer

from scrapy import signals
from scrapy.exceptions import NotConfigured, NotSupported
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.python import without_none_values


logger = logging.getLogger(__name__)


class DownloadHandlers:

    def __init__(self, crawler):
        self._crawler = crawler
        self._schemes = {}  # stores acceptable schemes on instancing | 存储实例化可接受的协议
        self._handlers = {}  # stores instanced handlers for schemes | 存储实例化可接受的处理函数
        self._notconfigured = {}  # remembers failed handlers | 存储失败的处理程序
        # 返回不为None的处理函数路径
        handlers = without_none_values(
            crawler.settings.getwithbase('DOWNLOAD_HANDLERS'))
        for scheme, clspath in handlers.items():
            # 实例化各个协议的处理函数
            self._schemes[scheme] = clspath
            self._load_handler(scheme, skip_lazy=True)
        # s.engine_stopped todo 这里有一个信号，暂时还不知道具体用处
        crawler.signals.connect(self._close, signals.engine_stopped)

    def _get_handler(self, scheme):
        """Lazy-load the downloadhandler for a scheme
        only on the first request for that scheme.
        仅在对该协议的第一个请求时才延迟加载该协议的下载处理程序。
        """
        # 注释的第一次请求才延迟再加是在init初始化就加载完成, 这里不会重复加载的意思
        if scheme in self._handlers:
            return self._handlers[scheme]
        if scheme in self._notconfigured:
            return None
        if scheme not in self._schemes:
            self._notconfigured[scheme] = 'no handler available for that scheme'
            return None

        return self._load_handler(scheme)

    def _load_handler(self, scheme, skip_lazy=False):
        path = self._schemes[scheme]
        try:
            # 将路径对应的类导入
            dhcls = load_object(path)
            if skip_lazy and getattr(dhcls, 'lazy', True):
                # 自定义懒加载或者类中自带这个属性，则跳过
                return None
            # 实例化
            dh = create_instance(
                objcls=dhcls,
                settings=self._crawler.settings,
                crawler=self._crawler,
            )
        except NotConfigured as ex:
            # 报错，则加入到未配置的协议
            self._notconfigured[scheme] = str(ex)
            return None
        except Exception as ex:
            logger.error('Loading "%(clspath)s" for scheme "%(scheme)s"',
                         {"clspath": path, "scheme": scheme},
                         exc_info=True, extra={'crawler': self._crawler})
            self._notconfigured[scheme] = str(ex)
            return None
        else:
            # 如果没报错，则加入到字典中，并返回实例
            self._handlers[scheme] = dh
            return dh

    def download_request(self, request, spider):
        # 这里就是真正的下载器了
        scheme = urlparse_cached(request).scheme
        # 利用合适的协议，找到合适的处理函数
        handler = self._get_handler(scheme)
        if not handler:
            raise NotSupported(f"Unsupported URL scheme '{scheme}': {self._notconfigured[scheme]}")
        return handler.download_request(request, spider)

    @defer.inlineCallbacks
    def _close(self, *_a, **_kw):
        for dh in self._handlers.values():
            if hasattr(dh, 'close'):
                yield dh.close()
