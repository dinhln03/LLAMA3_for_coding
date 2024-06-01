class PlayerResourceHand:
    def __init__(self):
        self.brick = 0
        self.grain = 0
        self.lumber = 0
        self.ore = 0
        self.wool = 0
        self.totalResources = 0

    def update(self):
        self.totalResources = self.brick + self.grain + self.lumber + self.ore + self.wool


class PlayerDevelopmentHand:
    def __init__(self):
        self.knights = 0
        self.roadBuildings = 0
        self.yearOfPlenty = 0
        self.monopolies = 0
        self.victoryPoints = 0
        self.totalDevelopments = 0

    def update(self):
        self.totalDevelopments = self.knights + self.roadBuildings + self.yearOfPlenty + self.monopolies \
                                 + self.victoryPoints


class EnemyPlayer:
    def __init__(self, turnOrder, name, color, nR, nS, nC, lR, lA, hS, dS, vVP):
        self.turnOrder = turnOrder
        self.name = name
        self.color = color
        self.handSize = hS
        self.developmentSize = dS
        self.visibleVictoryPoints = vVP
        self.numRoads = nR
        self.numSettlements = nS
        self.numCities = nC
        self.longestRoad = lR
        self.largestArmy = lA


class Player:
    def __init__(self, name, color, turnOrder):
        self.color = color
        self.name = name
        self.turnOrder = turnOrder
        self.numRoads = 15
        self.numSettlements = 5
        self.numCities = 4
        self.longestRoad = 0
        self.largestArmy = 0
        self.victoryPoints = 0
        self.resourceHand = PlayerResourceHand()
        self.developmentHand = PlayerDevelopmentHand()
        self.ownedRoads = list()
        self.ownedNodes = list()

    def getNumResources(self):
        return self.resourceHand.totalResources

    def getNumDevelopment(self):
        return self.developmentHand.totalDevelopments

    def getSendToEnemies(self):
        #  toSend = EnemyPlayer(self.turnOrder, self.name, self.color,
        #                      self.numRoads, self.numSettlements, self.numCities,
        #                      self.longestRoad, self.largestArmy)
        toSend = ','.join([self.turnOrder, self.name, self.color, self.numRoads, self.numSettlements, self.numCities,
                           self.longestRoad, self.largestArmy])
        return toSend

    def acquireRoad(self, road):
        self.ownedRoads.append(road)

    def acquireNode(self, node):
        self.ownedNodes.append(node)

    def addResources(self, array):
        self.resourceHand.brick += array[0]
        self.resourceHand.grain += array[1]
        self.resourceHand.lumber += array[2]
        self.resourceHand.ore += array[3]
        self.resourceHand.wool += array[4]
        self.resourceHand.totalResources += array[0] + array[1] + array[2] + array[3] + array[4]
