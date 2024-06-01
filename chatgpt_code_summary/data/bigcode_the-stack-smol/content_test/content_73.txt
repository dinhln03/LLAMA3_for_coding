import logging
from Models import Values
from classes import PlayerActions, OnePosition
from ActionsDispatcher import Action

class PositionsManagerDBException(RuntimeError):
  pass

class PositionsManager(object):
  def __init__(self, models, accountId, logger=None):
    self.models = models
    self.accountId = accountId
    if logger is None:
      logger = logging.getLogger()
    self.logger = logger
    self.restore()
  
  def restore(self):
    models = self.models
    profitThres = models.Values.get(Values.PositionThresProfit,
                                    accountId=self.accountId)
    if profitThres is None:
      raise PositionsManagerDBException('Settings "{k}" not initialized.'
                                        .format(k=Values.PositionThresProfit))
    self.profitThres = profitThres
    
    lossCutThres = models.Values.get(Values.PositionThresLossCut,
                                     accountId=self.accountId)
    if lossCutThres is None:
      raise PositionsManagerDBException('Settings "{k}" not initialized.'
                                        .format(k=Values.PositionThresLossCut))
    self.lossCutThres = lossCutThres
  
  @staticmethod
  def calcVariation(onetick, oneposition):
    """
    (tick: OneTick, position: OnePosition) -> float
    """
    created = oneposition.priceMean()
    if oneposition.side == OnePosition.SideLong:
      current = onetick.bid
    else:
      current = onetick.ask
    return current / created
  
  def makeDecision(self, positions):
    tick = self.models.Ticks.one()
    for p in positions:
      onePosition = p.positions[0]
      oneTick = tick[onePosition.exchanger]
      var = PositionsManager.calcVariation(oneTick, onePosition)
      if onePosition.side == OnePosition.SideLong:
        if var >= self.profitThres:
          return [(PlayerActions.CloseForProfit, p)] # Long, Profit
        elif var <= self.lossCutThres:
          return [(PlayerActions.CloseForLossCut, p)] # Long, LossCut
      else:
        if var <= 1.0 / self.profitThres:
          return [(PlayerActions.CloseForProfit, p)] # Short, Profit
        elif var >= 1.0 / self.lossCutThres:
          return [(PlayerActions.CloseForLossCut, p)] # Short, LossCut
    return []
  
  def getOpenPositions(self):
    return self.models.Positions.currentOpen(accountId=self.accountId)
  
  def createAction(self):
    positions = self.getOpenPositions()
    positions = filter(lambda p:p.isOpen(), positions)
    closes = self.makeDecision(positions)
    self.logger.debug('Completed decision, #close={n}.'.format(n=len(closes)))
    if len(closes) > 0:
      actionType, position = closes[0]
      return Action(actionType, position)
    else:
      return None
