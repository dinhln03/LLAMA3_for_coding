from backtest.indicators.StochRSI import StochRSI
import backtrader as bt
from backtest.indicators.ConnorsRSI import ConnorsRSI


class CRSI(bt.Strategy):
    params = (("ratio", 0.2),)

    def __init__(self) -> None:
        super().__init__()
        self.rsi = ConnorsRSI(self.data)

    def next(self):
        position = self.getposition()
        rsi = self.rsi[0]

        if not position and rsi >= 60:
            self.order_target_percent(target=self.p.ratio)

        if position and rsi <= 30:
            self.close()


class SRSI(bt.Strategy):
    params = (("ratio", 0.95),)

    def __init__(self) -> None:
        super().__init__()
        self.rsi = StochRSI(self.data, period=30)

        self.crossover = bt.indicators.CrossOver(self.rsi, bt.LineNum(0.7))
        self.crossdown = bt.indicators.CrossDown(self.rsi, bt.LineNum(0.3))

    def next(self):
        position = self.getposition()

        if not position and self.crossover[0]:
            self.order_target_percent(target=self.p.ratio)

        if position and self.crossdown[0]:
            self.close()
