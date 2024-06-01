#!/usr/bin/env python
# vim: set sts=4 sw=4 et:

import time
import xmlrpc.client
from . import players
from . import rpc
from .common import GameState, CardSet, GameError, RuleError, ProtocolError, simple_decorator
from .events import EventList, CardPlayedEvent, MessageEvent, TrickPlayedEvent, TurnEvent, StateChangedEvent

@simple_decorator
def error2fault(func):
    """
    Catch known exceptions and translate them to
    XML-RPC faults.
    """
    def catcher(*args):
        try:
            return func(*args)
        except GameError as error:
            raise xmlrpc.client.Fault(GameError.rpc_code, str(error))
        except RuleError as error:
            raise xmlrpc.client.Fault(RuleError.rpc_code, str(error))
        except ProtocolError as error:
            raise xmlrpc.client.Fault(ProtocolError.rpc_code, str(error))
    return catcher

@simple_decorator
def fault2error(func):
    """
    Catch known XML-RPC faults and translate them to
    custom exceptions.
    """
    def catcher(*args):
        try:
            return func(*args)
        except xmlrpc.client.Fault as error:
            error_classes = (GameError, RuleError, ProtocolError)
            for klass in error_classes:
                if error.faultCode == klass.rpc_code:
                    raise klass(error.faultString)

            raise error

    return catcher


class XMLRPCCliPlayer(players.CliPlayer):
    """
    XML-RPC command line interface human player.
    """
    def __init__(self, player_name):
        players.CliPlayer.__init__(self, player_name)
        self.game_state = GameState()
        self.hand = None

    def handle_event(self, event):
        if isinstance(event, CardPlayedEvent):
            self.card_played(event.player, event.card, event.game_state)
        elif isinstance(event, MessageEvent):
            self.send_message(event.sender, event.message)
        elif isinstance(event, TrickPlayedEvent):
            self.trick_played(event.player, event.game_state)
        elif isinstance(event, TurnEvent):
            self.game_state.update(event.game_state)
            state = self.controller.get_state(self.id)
            self.hand = state['hand']
            self.game_state.update(state['game_state'])
        elif isinstance(event, StateChangedEvent):
            self.game_state.update(event.game_state)
        else:
            print("unknown event: %s" % event)

    def wait_for_turn(self):
        """
        Wait for this player's turn.
        """
        while True:

            time.sleep(0.5)

            if self.controller is not None:
                events = self.controller.get_events(self.id)
                for event in events:
                    self.handle_event(event)

            if self.game_state.turn_id == self.id:
                break


class XMLRPCProxyController():
    """
    Client-side proxy object for the server/GameController.
    """
    def __init__(self, server_uri):
        super(XMLRPCProxyController, self).__init__()
        if not server_uri.startswith('http://') and \
            not server_uri.startswith('https://'):
            server_uri = 'http://' + server_uri

        self.server = xmlrpc.client.ServerProxy(server_uri)
        self.game_id = None
        self.akey = None

    @fault2error
    def play_card(self, _player, card):
        self.server.game.play_card(self.akey, self.game_id, rpc.rpc_encode(card))

    @fault2error
    def get_events(self, _player_id):
        return rpc.rpc_decode(EventList, self.server.get_events(self.akey))

    @fault2error
    def get_state(self, _player_id):
        state = self.server.game.get_state(self.akey, self.game_id)
        state['game_state'] = rpc.rpc_decode(GameState, state['game_state'])
        state['hand'] = rpc.rpc_decode(CardSet, state['hand'])
        return state

    @fault2error
    def player_quit(self, _player_id):
        self.server.player.quit(self.akey)

    @fault2error
    def register_player(self, player):
        player.controller = self
        plr_data = self.server.player.register(rpc.rpc_encode(player))
        player.id = plr_data['id']
        self.akey = plr_data['akey']

    @fault2error
    def start_game_with_bots(self):
        return self.server.game.start_with_bots(self.akey, self.game_id)

    @fault2error
    def create_game(self):
        self.game_id = self.server.game.create(self.akey)
        return self.game_id

