# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import messagebox
import ipaddress

from model import RecvTcpThread, RecvUdpThread, RecvMonitorThread, HachiUtil
from controller import LogController

# =================================
# == 定数
# =================================
DEF_PROTO = 1  # 0:TCP 1:UDP
DEF_DST_PROT = 12000

MAX_DATALEN = 9999

# =================================
# == 公開クラス
# =================================


class RecvParams(object):
    """ 受信パラメータ情報クラス """

    _instance = None

    def __new__(cls, *args, **keys):
        if cls._instance is None:
            cls._instance = object.__new__(cls)

            cls.proto = tk.IntVar(value=DEF_PROTO)
            cls.ip = tk.StringVar()
            cls.port = tk.IntVar(value=DEF_DST_PROT)

        return cls._instance


class MonitorParams(object):
    """ 受信モニター情報クラス """

    _instance = None

    def __new__(cls, *args, **keys):
        if cls._instance is None:
            cls._instance = object.__new__(cls)

            cls.datalen = tk.IntVar(value=0)
            cls.bps = tk.StringVar(value="0 bps")
            cls.pps = tk.IntVar(value=0)
            cls.recv_btn = tk.StringVar(value="受信開始")

            # "bps換算"動的更新
            cls.pps.trace_add('write', HachiUtil.UpdateBps(
                cls.datalen, cls.pps, cls.bps))

        return cls._instance


class RecvShareObject(object):
    """ スレッド間で値を共有するためのクラス """

    def __init__(self):
        self.count = 0
        self.total = 0


class RecvAction:
    def __init__(self, widgets):
        self.recvParams = RecvParams()
        self.monitorParams = MonitorParams()
        self.widgets = widgets
        self.stat = 0

        self.shareObj = RecvShareObject()

        # スレッド変数
        self.th_recv = None
        self.th_monitor = None

    def __call__(self, event=None):
        # 入力パラメータチェック
        msg = _param_check()
        if len(msg) > 0:
            messagebox.showwarning(title="warning", message=msg)
            return

        if self.stat == 0:
            self._recv_start()
        else:
            self._recv_stop()

    def _recv_start(self):
        # ログ出力インスタンス
        logger = LogController.LogController()

        # モニタースレッド開始
        self.monitor_start()

        # パケット受信スレッド開始
        # 0:TCP 1:UDP
        proto = RecvParams.proto.get()
        ip = RecvParams.ip.get()
        port = RecvParams.port.get()
        if proto == 0:
            logger.insert("TCPパケット受信を開始します({}:{})".format(ip, port))
            self.recv_tcp_start()
        elif proto == 1:
            logger.insert("UDPパケット受信を開始します({}:{})".format(ip, port))
            self.recv_udp_start()

        MonitorParams.recv_btn.set("受信停止")

        # ウィジェット非活性化
        for widget in self.widgets.values():
            widget.state(['disabled'])

        self.stat = 1

    def recv_tcp_start(self):
        """ TCPパケット送信スレッド """

        self.th_recv = RecvTcpThread.RecvTcpThread(RecvParams(), self.shareObj)
        self.th_recv.setDaemon(True)
        self.th_recv.start()

    def recv_udp_start(self):
        """ UDPパケット受信スレッド """

        self.th_recv = RecvUdpThread.RecvUdpThread(RecvParams(), self.shareObj)
        self.th_recv.setDaemon(True)
        self.th_recv.start()

    def monitor_start(self):
        """ パケット受信監視スレッド """

        self.th_monitor = RecvMonitorThread.RecvMonitorThread(
            MonitorParams(), self.shareObj)
        self.th_monitor.setDaemon(True)
        self.th_monitor.start()

    def _recv_stop(self):
        LogController.LogController().insert("パケット受信を停止します")
        """ スレッド停止 """

        # スレッド停止
        if self.th_recv is not None:
            self.th_recv.stop()
        if self.th_monitor is not None:
            self.th_monitor.stop()

        MonitorParams().recv_btn.set("受信開始")

        # 設定ウィジェット活性化
        for widget in self.widgets.values():
            widget.state(['!disabled'])

        self.stat = 0

# =================================
# == ローカル関数
# =================================


def _param_check():
    """ 受信パラメータチェック """

    msg = ""

    # IPアドレスチェック
    if not HachiUtil.LocalAddress().is_localaddress(RecvParams.ip.get()):
        # インタフェースなし
        msg += "・指定した待受IPアドレスがインターフェースにありません。\n"

    # ポート番号 0～65535
    if not (0 <= RecvParams.port.get() <= 65535):
        msg += "・ポート番号は 0～65535 の範囲で指定してください。\n"

    return msg
