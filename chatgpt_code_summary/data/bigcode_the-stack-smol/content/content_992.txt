# User class to hold name and __data

class User:

    ### Instance Variables ###
    __userName = ""
    __validUser = None
    __data = []

    __weights = []
    __notes = []
    __dates = []

    __intWeights = []

    __avgWeight = 0
    __minWeight = 0
    __maxWeight = 0
    ##########################

    ### Getters ###
    def getUserName(self):
        return self.__userName

    def getData(self):
        return self.__data

    def getValidUser(self):
        return self.__validUser

    def getWeights(self):
        return self.__weights

    def getNotes(self):
        return self.__notes

    def getDates(self):
        return self.__dates

    def getAvgWeight(self):
        return str(self.__avgWeight)

    def getMinWeight(self):
        return str(self.__minWeight)

    def getMaxWeight(self):
        return str(self.__maxWeight)
    ################

    ### Setters ###
    def setUserName(self, name):
        self.__userName = name

    def setData(self, data):
        self.__data = data

    def setValidUser(self, valid):
        self.__validUser = valid

    def setWeights(self, weights):
        self.__weights = weights

    def setNotes(self, notes):
        self.__notes = notes

    def setDates(self, dates):
        self.__dates = dates
    ################

    def addData(self, data):
        self.__data.append(data)

    def addWeight(self, weight):
        self.__weights.append(weight)

    def addNote(self, note):
        self.__notes.append(note)

    def addDate(self, date):
        self.__dates.append(date)

    def calcAvg(self):
        self.__avgWeight = int(sum(self.__intWeights)/len(self.__intWeights))

    def calcMaxWeight(self):
        self.__maxWeight = max(self.__intWeights)

    def calacMinWeight(self):
        self.__minWeight = min(self.__intWeights)

    def averageWeightDelta(self, weightData):
        pass

    def convertWeightList(self, weightData):
        for i in range(len(weightData)):
            weightData[i] = int(weightData[i])
        self.__intWeights = weightData
