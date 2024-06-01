# DADSA - Assignment 1
# Reece Benson

import json
from classes import Player as Player
from classes import Season as Season
from classes import Tournament as Tournament
from classes import Round as Round
from classes import Match as Match

class Handler():
    # Define the variables we will be using
    app = None
    prize_money = None
    seasons = { }

    def __init__(self, _app):
        if(_app.debug):
            print("[LOAD]: Loaded Handler!")

        # Define our Application within this Handler class
        self.app = _app

    # Used to load all data into memory
    def load(self):
        # This function will create our seasons and implement the genders & players
        self.load_prize_money()
        self.load_players()

        #TODO: Implement load_seasons()

    # Used to load prize money
    def load_prize_money(self):
        with open('./data/rankingPoints.json') as tData:
            data = json.load(tData)

            # Make our prize_money a dictionary
            if(self.prize_money == None):
                self.prize_money = { }

            # Make use of the values
            self.prize_money = [ (rank,pts) for pts in data for rank in data[pts] ]
        print(self.prize_money)

    # Used to load players from all seasons into memory
    def load_players(self):
        with open('./data/players.json') as tData:
            data = json.load(tData)

            # Players are classed within Seasons
            for season in data:
                # If the season does not yet exist, create it
                if(not season in self.seasons):
                    self.seasons[season] = { "players": { } }

                # Players are then stored within Gender classifications
                for gender in data[season]:
                    if(not gender in self.seasons[season]["players"]):
                        self.seasons[season]["players"][gender] = [ ]

                    # Append our player in the season, within the gender
                    for player in data[season][gender]:
                        #TODO: Change to using Player class
                        self.seasons[season]["players"][gender].append(player)

    def get_players(self, season):
        # Check our Season exists
        if(not season in self.seasons):
            return None
        else:
            # Check we have players within our Season
            if("players" in self.seasons[season]):
                return self.seasons[season]["players"]
            else:
                return None