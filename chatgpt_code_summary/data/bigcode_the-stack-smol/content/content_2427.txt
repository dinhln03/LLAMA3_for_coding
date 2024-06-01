"""This module contains the HelpCommandHandler class."""
from telegram import Update
from telegram.ext import CommandHandler, CallbackContext

import utils.helper as helper


class HelpCommandHandler(CommandHandler):
    """Handler for /help command"""

    def __init__(self):
        CommandHandler.__init__(self, "help", callback)


def callback(update: Update, _: CallbackContext):
    """Print the help text for a /start or /help command"""
    update.message.reply_text(helper.create_help_text())
