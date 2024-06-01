"""
config_objects.py
By: John-Michael O'Brien
Date: 7/25/2020

Data structures that define and load configuration information for the
wallpaper watcher.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import jsons
import yaml

@dataclass
class SubredditConfig():
    """ Holds any per-subreddit configuration. That's nothing right now. """

@dataclass
class MultiredditConfig():
    """ Holds information necessary to access a multireddit """
    user: str
    multi: str

@dataclass
class SourcesConfig():
    """ Holds information about image sources """
    subreddits: Optional[Dict[str, Optional[SubredditConfig]]]
    multis: Optional[Dict[str, MultiredditConfig]]

@dataclass
class Size():
    """ Holds a size """
    width: int
    height: int
    aspect_ratio: float = field(init=False, repr=False)
    def __post_init__(self):
        self.aspect_ratio = float(self.width) / float(self.height)

@dataclass
class TargetConfig():
    """ Holds information about a save target """
    path: str
    size: Size
    sources: List[str]
    allow_nsfw: bool = True

@dataclass
class WallpaperConfig():
    """ Loads and holds the configuration for wallpaperwatcher. """
    aspect_ratio_tolerance: float
    max_downloads: int
    update_interval: int
    sources: SourcesConfig
    targets: Dict[str, TargetConfig]

    @staticmethod
    def from_file(filename: str) -> "WallpaperConfig":
        """ Creates a WallpaperConfig from a YAML file """
        with open(filename, "r") as input_file:
            return jsons.load(yaml.load(input_file, Loader=yaml.SafeLoader), WallpaperConfig)

@dataclass
class RedditAuthInfo():
    """ Holds Reddit Authentication Values """
    client_id: str
    client_secret: str

    @staticmethod
    def from_file(filename: str) -> "RedditAuthInfo":
        """ Creates a RedditAuthInfo from a YAML file """
        with open(filename, "r") as input_file:
            auth = jsons.load(yaml.load(input_file, Loader=yaml.SafeLoader), RedditAuthInfo)
        return auth
