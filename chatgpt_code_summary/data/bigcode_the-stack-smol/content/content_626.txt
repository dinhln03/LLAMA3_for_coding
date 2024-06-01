from datetime import datetime

from chess_game.daos.player_dao import PlayerDao
from chess_game.models.player import Player


def test_player_dao_init(mongo_database):
    player_dao = PlayerDao(mongo_database)
    assert mongo_database == player_dao._mongo_database


def test_dao_create_and_find_player(mongo_database):
    start_date = datetime.now()
    player = Player(name="_Obi", stats={}, games=[], start_date=start_date)
    player_dao = PlayerDao(mongo_database)

    player_id = player_dao.create(player)
    loaded_player = player_dao.find_by_id(player_id)

    assert loaded_player['_id']
    assert "_Obi" == loaded_player['name']
    assert {} == loaded_player['stats']
    assert [] == loaded_player['games']
    assert f'{start_date:%Y-%m-%d %H:%M:%S}' == loaded_player['start_date']


def test_dao_create_and_find_players(mongo_database):
    player = Player()
    player_dao = PlayerDao(mongo_database)

    player_dao.create(player)
    player_id = player_dao.create(player)
    loaded_players = player_dao.find_all()

    assert len(loaded_players) > 1
    assert len([player for player in loaded_players if player_id == str(player['_id'])])
