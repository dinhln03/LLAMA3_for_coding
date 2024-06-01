#!usr/bin/env python3.7
#-*-coding:utf-8-*-

import json
import discord

PATH = "config.json"

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class Config:
    def __init__(self):
        with open(PATH,"r") as configfile:
            self.config = json.load(configfile)

        self.token = self.config["token"]
        self.owners = self.config["owner"]
        self.guildID = None
        if self.config["self-guild"].get("mode","load") == "load":
            self.guildID = self.config["self-guild"]["ID"]
        self.guildRegion = self.parseRegion(self.config["self-guild"]["region"])
        self.guild = None
        self.adminrole = None

    def __getitem__(self,item):
        return self.config[item]

    def initGuild(self, guild):
        self.guild = guild
        self.adminrole = discord.utils.get(self.guild.roles, name="Masakaki")

    @classmethod
    def parseRegion(cl, regionString):
        key = regionString.lower()
        if (key == "amsterdam"): return discord.VoiceRegion.amsterdam
        elif (key == "brazil"): return discord.VoiceRegion.brazil
        elif (key == "eu_central"): return discord.VoiceRegion.eu_central
        elif (key == "eu_west"): return discord.VoiceRegion.eu_west
        elif (key == "frankfurt"): return discord.VoiceRegion.frankfurt
        elif (key == "hongkong"): return discord.VoiceRegion.hongkong
        elif (key == "india"): return discord.VoiceRegion.india
        elif (key == "japan"): return discord.VoiceRegion.japan
        elif (key == "london"): return discord.VoiceRegion.london
        elif (key == "russia"): return discord.VoiceRegion.russia
        elif (key == "singapore"): return discord.VoiceRegion.singapore
        elif (key == "southafrica"): return discord.VoiceRegion.southafrica
        elif (key == "sydney"): return discord.VoiceRegion.sydney
        elif (key == "us_central"): return discord.VoiceRegion.us_central
        elif (key == "us_east"): return discord.VoiceRegion.us_east
        elif (key == "us_south"): return discord.VoiceRegion.us_south
        elif (key == "us_west"): return discord.VoiceRegion.us_west
        return None
