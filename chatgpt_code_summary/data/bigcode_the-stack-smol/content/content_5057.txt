"""
Module for handling the custom Lua commands for the bot
"""

import lupa
import shlex
import time
from threading import Thread
from .utils import human_readable_time, ArgumentParser
from .http import Http, TupleData
from .timer import Interval, Delayed
from .chat import Chat


class CommandPermissionError(BaseException):
    """
    An exception that happens when a user tries to execute a custom command
    without the appropriate user level for it.
    """
    pass

class CommandCooldownError(BaseException):
    """
    An exception that happens when a user tries to execute a custom command
    during it's cooldown period.
    """
    pass


class DataSource(object):
    """
    A simple structure to allow Lua to store and read data from the database

    Call from Lua via the injected _G["datastore"] instance:

    .. code-block:: lua

        _G["datastore"].set("my-data", "my-value")
        _G["datastore"].get("my-data")

    If not working directly with the datasource implementation you should
    however use the datasource wrapper:

    .. code-block:: lua

        local ds = require('datasource')
        ds.set("my-data", "my-value")
        ds.get("my-data")
    """

    def __init__(self, channel, bot, data=None):
        if not data:
            data = {}

        self.channel = channel
        self.bot = bot
        self.data = data

    def get(self, key):
        """
        Get a single value from the database

        :param key: The name of the value
        :return: The stored value or "null" if not found
        """

        if not key in self.data:
            # TODO: Check how to work around this silly shit
            return "null"

        return self.data[key]

    def set(self, key, value):
        """
        Set a single value to the database

        :param key: The name of the value
        :param value: The value to be stored
        :return: None
        """

        self.data[key] = value
        self.bot.update_global_value(self.channel, key, value)


class CommandManager(object):
    """
    Manager for custom commands
    """

    # Template for creating new Lua functions
    func_template = u"""
    function __chat__{func_name}({args})
        {func_body}
    end
    """

    # Function template for doing Lua function calls
    # {{ and }} are escaped { and } for .format()
    call_template = u"""
    function(...)
        local chat = require("chat")
        local retval = __chat__{func_name}(unpack(table.pack(...)))
        if retval ~= nil then
            chat.message(retval)
        end

        return retval
    end
    """

    def __init__(self, channel, bot, settings=None, data=None, logger=None,
                 chat=None):

        self.channel = channel
        self.bot = bot

        if chat:
            self.chat = chat
        else:
            self.chat = Chat(self.bot, self.channel)

        self.settings = settings
        self.logger = logger
        self.commands = {}
        self.timers = []
        self.datasource = DataSource(channel, bot, data)
        self.commands_last_executed = {}

        self.lua = lupa.LuaRuntime(unpack_returned_tuples=False)
        self._inject_globals()

    def stop_timers(self):
        """
        Cancel all timers still running

        :return:
        """

        for timer in self.timers:
            timer.cancel()

    def add_command(self, args):
        """
        Handler for the "def" -commands in chat

        :param args: All the words after the "def" -command
        :return: The created command, if any, and the minimum required user
                 level
        """

        added, command, flags, user_level, code = self._parse_func(args)

        channel, command, flags, user_level, code = self.load_command(
            command, flags, user_level, code
        )

        return added, channel, command, flags, user_level, code

    def add_simple_command(self, args):
        """
        Handler for the "com" -commands in chat

        :param args: All the words after the "com" -command
        :return: A bunch of stuff
        """

        added, command, flags, user_level, code = self._parse_simple_func(args)

        channel, command, flags, user_level, code = self.load_command(
            command, flags, user_level, code
        )

        return added, channel, command, flags, user_level, code


    def is_valid_command(self, command):
        """
        Check if the given command is registered

        :param command: The name of the command
        :return: True or False
        """

        return command in self.commands

    def load_command(self, command, flags, user_level, code, set=True):
        """
        Load a command in the runtime

        :param command: What is the command called
        :param flags: Command flags
        :param user_level: The minimum user level to run the command
        :param code: The Lua code for the custom command
        :param set: Should the command be set on the bot via set_command,
                    set this to False when loading commands from e.g. the
                    database
        :return: None
        """

        if self.logger:
            self.logger.debug(u"Loading command {0} with user level "
                              u"{1}".format(
                command, user_level
            ))

        self.commands[command] = {
            "flags": flags,
            "user_level": user_level,
            "code": code
        }

        self.load_lua(code)

        return self.channel, command, flags, user_level, code

    def run_command(self, nick, user_level, command, args=None,
                    timestamp=None, threaded=True):
        """
        Handles running of custom commands from chat

        :param nick: The calling user
        :param user_level: The calling user's level
        :param command: The command triggered
        :param args: The words on the line after the command
        :param timestamp: The unixtime for when the event happened
        :return: Any return value from the custom Lua command, to be sent
                 back to the channel
        :raise CommandPermissionError: If user lacks permissions for command
        """

        if not self._can_run_command(user_level, command):
            raise CommandPermissionError(u"User does not have permission to "
                                         u"run this command")

        if args is None:
            args = []
        else:
            if "quoted" in self.commands[command]["flags"]:
                if self.commands[command]["flags"]["quoted"] == 1:
                    text = " ".join(args)
                    args = shlex.split(text)

        if timestamp is None:
            timestamp = time.time()

        if self._is_under_cooldown(command, timestamp):
            raise CommandCooldownError()

        self._set_last_executed_time(command, timestamp)

        def run():
            code = self.call_template.format(func_name=command)
            lua_func = self.lua.eval(code)
            if "want_user" in self.commands[command]["flags"]:
                if self.commands[command]["flags"]["want_user"] == 1:
                    args.insert(0, nick)

            return lua_func(*args)

        if threaded:
            lua_thread = Thread(target=run)
            lua_thread.daemon = True
            lua_thread.start()
        else:
            return run()

    def load_lua(self, code):
        """
        Load Lua code in our runtime

        :param code: The Lua code
        :return: None
        """

        self.lua.execute(code)

    def _parse_func(self, args):
        """
        Process the given arguments into a function definition

        :param args: List of the words after the "def" command
        :return: Function name, if it wants the caller's user name,
                 the required user level, and the function's Lua code
        :raise argparse.ArgumentError: There was something wrong with the args
        """

        parser = ArgumentParser()
        parser.add_argument("-ul", "--user_level", default="mod")
        parser.add_argument("-c", "--cooldown", default=None)
        parser.add_argument("-a", "--args", default="")
        parser.add_argument("-w", "--want_user", action="store_true",
                            default=False)
        parser.add_argument("-q", "--quoted", action="store_true",
                            default=False)
        parser.add_argument("func_name")
        parser.add_argument("func_body", nargs='*')

        options = parser.parse_args(args)

        # Rebuild code

        if options.want_user:
            new_args = "user"
            if len(options.args) > 0:
                new_args += ","

            options.args = new_args + options.args

        code = self.func_template.format(
            func_name=options.func_name,
            args=options.args,
            func_body=" ".join(options.func_body)
        )

        flags = {
            "want_user": int(options.want_user),
            "quoted": int(options.quoted),
            "cooldown": (int(options.cooldown) if options.cooldown else None)
        }

        added = bool(options.func_body)

        return added, options.func_name, flags, options.user_level, code

    def _parse_simple_func(self, args):
        """
        Process the given arguments into a simple function definition

        :param args: List of the words after the "com" command
        :return: Function name, if it wants the caller's user name,
                 the required user level, and the function's Lua code
        :raise argparse.ArgumentError: There was something wrong with the args
        """

        parser = ArgumentParser()
        parser.add_argument("-ul", "--user_level", default="mod")
        parser.add_argument("-c", "--cooldown", default=None)
        parser.add_argument("func_name")
        parser.add_argument("response_text", nargs='*')

        options = parser.parse_args(args)

        # Rebuild response
        response_text = " ".join(options.response_text)
        response_text = response_text.replace("\\", "\\\\")
        response_text = response_text.replace('"', '\\"')

        func_body = u"""
        return SimpleCom("{response_text}", user, table.pack(...))
        """.format(response_text=response_text)

        code = self.func_template.format(
            func_name=options.func_name,
            args="user,...",
            func_body=func_body
        )

        flags = {
            "want_user": 1,
            "quoted": 0,
            "cooldown": (int(options.cooldown) if options.cooldown else None)
        }

        added = bool(options.response_text)

        return added, options.func_name, flags, options.user_level, code

    def _is_under_cooldown(self, command, timestamp):
        """
        Check if this command's cooldown period is in effect
        :param command: Which command
        :param timestamp: What is the timestamp it was issued on
        :return:
        """

        if command in self.commands_last_executed:
            if "cooldown" in self.commands[command]["flags"]:
                cooldown_period = self.commands[command]["flags"]["cooldown"]
                last_executed = self.commands_last_executed[command]

                if cooldown_period is not None:
                    cooldown_expires = last_executed + cooldown_period

                    if timestamp < cooldown_expires:
                        return True

        return False

    def _set_last_executed_time(self, command, timestamp):
        """
        Save the last execution time of a command
        :param command: Which command
        :param timestamp: What is the timestamp it was issued on
        :return:
        """
        self.commands_last_executed[command] = timestamp

    def _level_name_to_number(self, name):
        """
        Convert the given user level to a number

        :param name: Level name
        :return: A number between 0 and Infinity, higher number is higher
                 user level
        :raise ValueError: In case of invalid user level
        """

        levels = [
            "user",
            "reg",
            "mod",
            "owner"
        ]

        if not name in levels:
            raise ValueError(u"{0} is not a valid user level".format(name))

        return levels.index(name)

    def _can_run_command(self, user_level, command):
        """
        Check if this command can be run with the given user level

        :param user_level: The calling user's level
        :param command: The command being called
        :return: True of False
        """

        need_level = self._level_name_to_number(
            self.commands[command]["user_level"]
        )

        got_level = self._level_name_to_number(user_level)

        return got_level >= need_level

    def _inject_globals(self):
        """
        Inject some Python objects and functions into the Lua global scope _G

        :return: None
        """

        injector = self.lua.eval("""
            function (key, value)
                _G[key] = value
            end
        """)

        def log(message):
            """
            Pass a message from Lua to the Python logger

            :param message: The message text
            :return: None
            """

            self.logger.debug(u"Lua: " + str(message))

        def interval(seconds, function):
            i = Interval(seconds, function, self.lua)
            self.timers.append(i)
            return i

        def delayed(seconds, function):
            i = Delayed(seconds, function, self.lua)
            self.timers.append(i)
            return i

        def simple_com(text, user, args):
            params = []
            if args:
                for key in args:
                    if key != "n":
                        params.append(args[key])

            try:
                response = text.format(*params, user=user)
            except IndexError:
                response = user + u", invalid number of arguments."

            return response

        injector("log", log)
        injector("datasource", self.datasource)
        injector("human_readable_time", human_readable_time)
        injector("settings", self.settings)
        injector("Chat", self.chat)
        injector("Http", Http())
        injector("TupleData", TupleData)
        injector("Interval", interval)
        injector("Delayed", delayed)
        injector("SimpleCom", simple_com)
