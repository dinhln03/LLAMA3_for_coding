import pandas as pd
from math import exp, log,sqrt
from numpy import cumsum,std,sum, mean

def outData(ts,actionHist,indx,startIndex=0):
    out=pd.DataFrame(ts,index=indx,columns=['ts']).applymap(lambda x: x/100)
    out=out[startIndex:]
    out['cum_log_ts']=cumsum([log(1+i) for i in out['ts']])
    out['Action_Hist']=actionHist[startIndex:]
    out['trading rets']=calculateTradingReturn(out['Action_Hist'],out['ts'])
    out['cum_log_rets']=cumsum([log(1+x) for x in out['trading rets']])
    return out

def calculateTradingReturn(actionHistory,tsReturn,delta=0):
    if ((type(tsReturn)==pd.core.frame.DataFrame) or (type(tsReturn)==pd.core.frame.Series)):
        rets=pd.Series(index=tsReturn.index)
    else:
        rets=[0 for i in range(len(tsReturn))]
    for t in range(len(tsReturn)-1):
        cost=delta*abs(actionHistory[t+1]-actionHistory[t])
        rets[t]=(1+(actionHistory[t]*tsReturn[t]))*(1-cost)-1
    return rets

def maximumDrawdown(ts):
    return min(ts)

def annualisedSharpe(rs,rf=0):
    rs=rs[:-1]
    if (type(rf)==int)|(type(rf)==float):
        rf=[rf for i in rs]
    mean_ann_ret=mean([(rs[i]*252)-rf[i] for i in range(len(rs))])
    stand= std(rs)*sqrt(252)
    return (mean_ann_ret)/stand

def percentOfOutperformedMonths(tradingRets,tsRets):
    monthlyTrating=tradingRets.resample('M').apply(logCumSum)
    monthlyMkt=tsRets.resample('M',how=logCumSum)
    numOutperform=0
    for i in range(len(monthlyMkt)):
        if monthlyTrating[i]>monthlyMkt[i]:
            numOutperform+=1
    return 100*((1.0*numOutperform)/len(monthlyMkt))

def numTradesPerYear(actionHistory):
    count=0
    for i in range(1,len(actionHistory)):
        if actionHistory[i]!=actionHistory[i-1]:
            count+=1
    return count/252

def totalReturn(log_returns):
    return exp(sum(log_returns+1))-1

def logCumSum(ts):
    return sum([log(1+t) for t in ts])
    pass
