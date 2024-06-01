# name  : Shoby Gnanasekaran
# net id: shoby

from dungeonchar import DungeonCharacter
from healable import Healable
from hero import Hero

class Priestess(Hero, Healable):
    """ Priestess is a hero with it own statistics. The basic behaviour is same as the hero.
    Special ability is to heal everytime after taking damage """
    def __init__(self, name, model, **kwargs):
        super().__init__(name = name, model = model, **kwargs)
        super(DungeonCharacter, self).__init__(**kwargs)

    def take_damage(self, dmg, source):
        """ after taking damage, if the priestess is not dead, it heals itself"""
        hp_before_attack = self.hp
        super().take_damage(dmg, source)
        if self._is_alive and hp_before_attack > self.hp and source != "pit":
            heal_message = self.heal_itself()
            self.model.announce(f"{self.name}: {heal_message}")



