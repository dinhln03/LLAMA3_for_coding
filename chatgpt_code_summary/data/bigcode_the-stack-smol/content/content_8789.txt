import AnalysisModule as Ass
import GraphFunctions as Gfs
import yfinance as yf
import DatabaseStocks as Ds

listOfStocksToAnalyze = Ds.get_lists()

macdproposedbuylist = []
macdproposedselllist = []
proposedbuylist = []
proposedselllist = []

for stock in listOfStocksToAnalyze:
    # print(stock)
    StockData = yf.Ticker(stock).history(period="1y")
    if Ass.macd_potential_buy(StockData) and Ass.is_stock_rising(StockData):
        macdproposedbuylist.append(stock)
        print("MACD Something you might wanna buy is " + stock)
        continue

    if Ass.macd_potential_sell(StockData) and Ass.is_stock_falling(StockData):
        macdproposedselllist.append(stock)
        print("MACD Something you might wanna sell is " + stock)

    if Ass.sma_potential_buy(StockData):
        proposedbuylist.append(stock)
        print("Something you might wanna buy is " + stock)
        continue

    if Ass.sma_potential_sell(StockData):
        proposedselllist.append(stock)
        print("Something you might wanna sell is " + stock)

print(macdproposedselllist)
print(macdproposedbuylist)

for stock in macdproposedbuylist:
    StockData = yf.Ticker(stock).history(period="1y")
    Gfs.draw_macd_buy(StockData, "BUY " + stock)

for stock in macdproposedselllist:
    StockData = yf.Ticker(stock).history(period="1y")
    Gfs.draw_macd_sell(StockData, "SELL " + stock)

for stock in proposedbuylist:
    StockData = yf.Ticker(stock).history(period="1y")
    Gfs.draw_macd_buy(StockData, "BUY MA " + stock)

for stock in proposedselllist:
    StockData = yf.Ticker(stock).history(period="1y")
    Gfs.draw_macd_sell(StockData, "SELL MA " + stock)
