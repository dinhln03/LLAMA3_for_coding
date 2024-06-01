# Columbus - A Smart Navigation System for the Visually-Impaired
# Ike Kilinc

# This file integrates Columbus' primary start location and destination input
# features with its core pathfinding algorithm. This file also facilitates
# Columbus' speech recognition and audio functionalities.

from speech_to_text import *
from node_mapper import *
from path_finder import *

#####################################################################
#####################################################################

def run():
    # Columbus asks what the user would like to do (with help option). directions, popular dests, directions
    pathMode = startupModeSelection()

    if pathMode == "specificDestination":
        # User inputs destination.
        destination = destinationInput()
        startLocation = startLocationInput()

    elif pathMode == "nearestRestroom":
        # Columbus asks where user is (TEMP).
        startLocation = startLocationInput()
        # Columbus finds nearest Restroom and sets as destination
        destination = None

    elif pathMode == "nearestPrinter":
        # Columbus asks where user is (TEMP).
        startLocation = startLocationInput()
        # Columbus finds nearest Printer and sets as destination
        destination = None

    elif pathMode == "popularDestinations":
        # Columbus gives user choice options of popular destinations.
        # Sets user input as the destination.
        destination = popularLocationsInput(data)
        startLocation = startLocationInput()

    elif pathMode == "savedDestinations":
        # Columbus gives user choice of previously saved destinations and sets
        # user input as the destination.
        destination = savedLocationsInput(data)
        startLocation = startLocationInput()

    elif pathMode == "findGod":
        pass

    # Columbus searches for and determines path to destination.
    nodesPath = pathFinder(startLocation, destination, pathMode)

#####################################################################
#####################################################################

class Segment(object):
    def __init__(self, startCoords, endCoords, segNumber, isActive, isFloorChange):
        self.segmentBounds = (startCoords[0], startCoords[1], endCoords[0], endCoords[1])
        self.floor = startCoords[2]
        self.segNumber = segNumber
        self.isActive = isActive
        self.isFloorChange = isFloorChange
        # self.direction = direction
    def __repr__(self):
        return str(self.segNumber)
    def __hash__(self):
        return hash(self.segNumber)
    def getSegBounds(self):
        return self.segmentBounds
    def getSegNum(self):
        return self.segNumber
    def getSegFloor(self):
        return self.floor
    def getIsActive(self):
        return self.isActive
    def getIsFloorChange(self):
        return self.isFloorChange
    def getCenter(self):
        centerX = (self.segmentBounds[0] + self.segmentBounds[2])/2
        centerY = (self.segmentBounds[1] + self.segmentBounds[3])/2
        return (centerX, centerY)
    def getSegmentDirection(self):
        (x0,y0,x1,y1) = self.segmentBounds
        if (x1-x0) > 0: return "E"
        elif (x1-x0) < 0: return "W"
        elif (y1-y0) > 0: return "S"
        elif (y1-y0) < 0: return "N"
        else: return None


def createAllSegments(nodesPath):
    allSegments = []
    isFloorChange = False

    intNodesPath = []
    for i in range(len(nodesPath)):
        node = nodesPath[i]
        if (isinstance(node, Intersection) or isinstance(node, Elevator) or
            i==0 or i==(len(nodesPath)-1)):
            intNodesPath.append(node)

    for i in range(len(intNodesPath)-1):
        (node, nextNode) = (intNodesPath[i], intNodesPath[i+1])
        
        if (isinstance(node, Elevator) and isinstance(nextNode, Elevator)):
            isFloorChange = True

        segment = Segment(node.getCoords(), nextNode.getCoords(), i, False, isFloorChange)
        isFloorChange = False

        allSegments.append(segment)

    allSegments.append(Segment(intNodesPath[-1].getCoords(), intNodesPath[-1].getCoords(), i, False, False))
    return allSegments


#####################################################################
#####################################################################

def startupModeSelection(repeat=False):
    # Used to select mode for operating Columbus. Mode options include:
    # Finding directions to a specific destination, directions to the nearest
    # restroom, directions to popular destinations, and directions to previously
    # saved destinations.
    if repeat == True:
        play("voiceCommands/sorryPleaseRepeat.wav")
    else:
        play("voiceCommands/modeSelectionInputPrompt.wav")

    userInput = recognizeSpeech("mode")

    if userInput == "help":
        play("voiceCommands/modeSelectionHelp.wav")

        userInput = recognizeSpeech("mode")

    if userInput in ["nearestRestroom", "popularDestinations",
                     "savedDestinations", "nearestPrinter",
                     "specificDestination", "findGod", "help"]:
        return userInput

    else:
        return startupModeSelection(True)


def destinationInput(repeat=False):
    if repeat==True:
        play("voiceCommands/sorryPleaseRepeat.wav")
    else:
        # Columbus asks where user would like to go.
        play("voiceCommands/destinationInputPrompt.wav")

    # User inputs destination
    destination = recognizeSpeech("location")

    if isLegalNode(destination):
        return destination
    else:
        return destinationInput(True)


def startLocationInput(repeat=False):
    if repeat==True:
        play("voiceCommands/sorryPleaseRepeat.wav")
    else:
        # Columbus asks where user is now.
        play("voiceCommands/startLocationInputPrompt.wav") 

    # User inputs start location.
    startLocation = recognizeSpeech("location")

    if isLegalNode(startLocation):
        return startLocation
    else:
        return startLocationInput(True)


def popularLocationsInput(data, repeat=False):
    print("popLocsInput")
    if repeat==True:
        play("voiceCommands/sorryPleaseRepeat.wav")
    else:
        # Columbus asks where user would like to go.
        play("voiceCommands/destinationInputPromptWithHelp.wav")

    userInput = recognizeSpeech("popularDest")

    if userInput == "help":
        play("voiceCommands/popularLocationSelectionHelp.wav")
        userInput = recognizeSpeech("popularDest")

    if userInput in ["5Prima", "4Sorrells"]:
        return userInput

    else:
        return popularLocationsInput(data, True)


def savedLocationsInput(data, repeat=False):
    if len(data.savedLocations) == 0:
        play("voiceCommands/noSavedDestinations.wav")

    else:
        if repeat==True:
            play("voiceCommands/sorryPleaseRepeat.wav")
        else:
            # Columbus asks where user would like to go.
            play("voiceCommands/destinationInputPromptWithHelp.wav")

        userInput = recognizeSpeech("savedDest")

        if userInput == "help":
            play("voiceCommands/modeSelectionHelp.wav")
            userInput = recognizeSpeech("savedDest")

        if userInput in data.savedLocations:
            return userInput

        else:
            return savedLocationsInput(data, True)


def isLegalNode(string):
    allNodesMap = mapAllNodes()

    for floor in allNodesMap:
        for roomStr in allNodesMap[floor]:
            if string == roomStr:
                return True
    return False



