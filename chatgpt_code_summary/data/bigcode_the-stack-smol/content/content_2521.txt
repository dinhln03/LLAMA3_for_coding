# -*- coding: utf-8 -*-
import time, threading, uuid, sys
import tushare as ts
from PyQt4 import QtCore, QtGui
import utils

class ProfitStrategy(QtCore.QObject):
    def init(self, b):
        pass
    def update_target(self, dp, p, t1, t2):
        pass
    def reset_target(self, b, p, t1, t2):
        pass

class ProfitWideStrategy(QtCore.QObject):
    def init(self, b):
        dp = b
        t1 = dp * 1.08
        t2 = dp * 1.12
        p = dp * 1.06
        return (dp, p, t1, t2)
    def update_target(self, dp, p, t1, t2):
        dp = t1
        t1 = dp * 1.08
        t2 = dp * 1.12
        p = dp * 1.06
        return (dp, p, t1, t2)
    def reset_target(self, dp, p, t1, t2):
        t1 = dp
        dp = t1 / 1.08
        p = dp * 1.06
        t2 = dp * 1.12
        return (dp, p, t1, t2)

class ProfitThinStrategy(QtCore.QObject):
    def init(self, b):
        dp = b
        t1 = dp * 1.08
        t2 = dp * 1.12
        p = dp * 1.06
        return (dp, p, t1, t2)
    def update_target(self, dp, p, t1, t2):
        t1 = t2
        dp = t1 / 1.08
        p = dp * 1.06
        t2 = p * 1.12
        return (dp, p, t1, t2)
    def reset_target(self, dp, p, t1, t2):
        t2 = t1
        dp = t2 / 1.08
        p = dp * 1.06
        t1 = dp * 1.12
        return (dp, p, t1, t2)

class SaveProfit(QtCore.QObject):
    _saveProfitSignal = QtCore.pyqtSignal(int)
    _resetSignal = QtCore.pyqtSignal(int)
    _targetSignal = QtCore.pyqtSignal(int, int)

    def __init__(self, id, base_cost, strategy=ProfitWideStrategy()):
        super(SaveProfit, self).__init__()
        self._strategy = strategy
        self._id = id
        self._trigger_count = 0
        self._trigge_target = False
        self._base_cost = base_cost
        self._dynamic_cost, self._profit, self._target1, self._target2 = \
            self._strategy.init(self._base_cost)

    def run(self, price):
        self._temp_price = price
        if self._trigge_target:
            if price >= self._target2:
                self._trigge_target = False
                self._trigger_count += 1
                self._dynamic_cost, self._profit, self._target1, self._target2 = \
                    self._strategy.update_target(self._dynamic_cost, self._profit, self._target1, self._target2)
                self._targetSignal.emit(self._id, self._trigger_count)
            elif price < self._profit:
                #warning
                print self.info()
                self._saveProfitSignal.emit(self._id)
                return False
            elif price >= self._profit:
                if self._base_cost > self._profit and price >= self._base_cost:
                    self._resetSignal.emit(self._id)
                    self._trigge_target = False
                    self._dynamic_cost, self._profit, self._target1, self._target2 = \
                        self._strategy.update_target(self._dynamic_cost, self._profit, self._target1, self._target2)
        else:
            last_profit = self._dynamic_cost / 1.08 * 1.06
            if price >= self._target1:
                self._trigge_target = True
            elif price <= self._dynamic_cost:
                self._trigge_target = True
                self._trigger_count -= 1
                self._dynamic_cost, self._profit, self._target1, self._target2 = \
                    self._strategy.reset_target(self._dynamic_cost, self._profit, self._target1, self._target2)
        return True

    def info(self):
        return {
            "dyprice" : self._dynamic_cost,
            "target1" : self._target1,
            "target2" : self._target2,
            "profit" : self._profit,
            "base" : self._base_cost,
            "cur" : self._temp_price,
            "trigged" : self._trigge_target,
            "trigger_count" : self._trigger_count
        }

class StcokWatcher(QtCore.QObject):
    def __init__(self, stock_infos):
        super(StcokWatcher, self).__init__()
        self._stock_infos = stock_infos #code,price,name, triggered
        self._on_watch = False
        self._t = threading.Thread(target=self.on_watch)
        self._t.setDaemon(True)

    def init(self):
        self._profiters = []
        self._stocks = []
        for i in range(len(self._stock_infos)):
            stock_info = self._stock_infos[i]
            self._stocks.append(stock_info['code'])
            base_price = stock_info['base']
            if (stock_info.has_key('stragegy') and stock_info['stragegy'] == 1):
                profiter = SaveProfit(i, base_price, ProfitThinStrategy())
            else:
                profiter = SaveProfit(i, base_price)
            self._profiters.append(profiter)
            self._profiters[i]._saveProfitSignal.connect(self.on_warn)
            self._profiters[i]._resetSignal.connect(self.on_reset)
        df = ts.get_realtime_quotes(self._stocks)
        for i in df.index:
            quote = df.loc[i]
            self._stock_infos[i]['name'] = (quote['name'])

    def on_watch(self):
        while self._on_watch:
            df = ts.get_realtime_quotes(self._stocks)
            print '-' * 30
            print "股票名  触发  当前价格  成本价格  收益点  收益率  触发次数"
            for i in df.index:
                quote = df.loc[i]
                self._profiters[i].run(float(quote['price']))
                #print self._profiters[i].info()
                info = self._profiters[i].info()
                prate = (info["cur"] - info["base"]) * 100 / info["cur"]
                prate = int(prate)]
                triggerstr = '是' if info['trigged'] else '否'
                print "%s %s %8.3f %8.3f %8.3f %8d%% %8d" % \
                      (self._stock_infos[i]['name'], triggerstr, info['cur'], info['base'], info['profit'], prate, info['trigger_count'])
                #print info
            time.sleep(3)

    def on_warn(self, id):
        #return
        __business_id = uuid.uuid1()
        profiter = self._profiters[id].info()
        stock_info = self._stock_infos[id]
        prate = (profiter["cur"] - profiter["base"]) * 100 /  profiter["cur"]
        prate = int(prate)
        params = "{\"nm\":\"%s\",\"number\":\"%s\",\"in\":\"%.3f\",\"cur\":\"%.3f\",\"prate\":\"%d%%\"}"  \
                 % (stock_info['name'], stock_info['code'], profiter["base"], profiter["cur"], prate)

        if not stock_info.has_key('msg') or not stock_info['msg']:
            print '+' * 40
            print utils.send_sms(__business_id, "13564511106", "XK咨询", "SMS_94650115", params)
            print '+' * 40
            stock_info['msg'] = True

    def on_reset(self, id):
        self._stock_infos[id]['msg'] = False

    def start(self):
        self._on_watch = True
        self._t.start()

if __name__ == "__main__":
    stocks = [
        {'code':'600516', 'base':34.313,'stragegy':1},  # 方大碳素
        {'code':'002145', 'base':6.682},  # 中核钛白
        {'code':'603079', 'base':69.819},  # 盛大科技
        {'code':'002888', 'base':35.119},  # 惠威科技
        {'code':'603826', 'base':20.609}  # 坤彩科技
    ]

    qApp = QtGui.QApplication(sys.argv)
    watchers = StcokWatcher(stocks)
    watchers.init()
    watchers.start()
    qApp.exec_()