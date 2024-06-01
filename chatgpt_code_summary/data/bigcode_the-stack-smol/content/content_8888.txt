from copy import deepcopy

from quest.quest_manager import QuestManager
import settings
from twitch.channel import Channel


class QuestChannel(Channel):
    def __init__(self, owner, channel_manager):
        super().__init__(owner, channel_manager)

        self.quest_manager = QuestManager(self)

        self.mod_commands.add_commands(
            exact_match_commands={
                '!queston': lambda _: self.channel_manager.enable_quest(self.owner),
                '!questoff': lambda _: self.channel_manager.disable_quest(self.owner)
            }, starts_with_commands={
                '!questcooldown': self.set_quest_cooldown})

    def set_quest_cooldown(self, display_name, cooldown):
        """
        Sets the quest cooldown to be the specified value.

        :param display_name: str - The display name of the person trying to set the cooldown
        :param cooldown: str - The raw message specifying the value to set the cooldown to
        :return:
        """
        try:
            self.channel_manager.set_quest_cooldown(self.owner, int(cooldown))
        except (IndexError, ValueError):
            self.channel_manager.bot.send_whisper(
                display_name, 'Invalid usage! Sample usage: !questcooldown 90')

    def check_commands(self, display_name, msg, is_mod, is_sub):
        """
        Connect to other command lists whose requirements are met.
        :param display_name: str - The display name of the command sender
        :param msg: str - The full message that the user sent that starts with "!"
        :param is_mod: bool - Whether the sender is a mod
        :param is_sub: bool - Whether the sender is a sub
        """
        super().check_commands(display_name, msg, is_mod, is_sub)

        # Check quest commands
        self.quest_manager.commands.execute_command(display_name, msg)
