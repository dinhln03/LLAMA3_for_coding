import berserk
import chaturanga

token = 'token'
bot_id = 'sultankhan2'

session = berserk.TokenSession(token)
lichess = berserk.Client(session)

for event in lichess.bots.stream_incoming_events():
    if event['type'] == 'challenge':
        challenge = event['challenge']
        if challenge['variant']['key'] == 'standard':
            if not challenge['rated']:
                game_id = challenge['id']
                lichess.bots.accept_challenge(game_id)
    else:
        game_id = event['game']['id']
        challenge = {'color': 'random'}

    for game_state in lichess.bots.stream_game_state(game_id):
        if game_state['type'] == 'gameFull':
            if game_state['state']['moves'] == '':
                if game_state['initialFen'] == 'startpos':
                    Chessboard = chaturanga.Chessboard()
                else:
                    Chessboard = chaturanga.Chessboard(
                        game_state['initialFen'])
                if challenge['color'] == 'random':
                    if 'id' in game_state['white']:
                        is_white = game_state['white']['id'] == bot_id
                    else:
                        is_white = False
                else:
                    is_white = {
                        'white': False,
                        'black': True
                    }[challenge['color']]

            if is_white:
                bot_move = chaturanga.bot(Chessboard)[0]
                Chessboard.move(bot_move)
                lichess.bots.make_move(game_id, bot_move)

        if game_state['type'] == 'gameState':
            moves = game_state['moves'].split(' ')
            if len(moves) % 2 != is_white:
                Chessboard.move(moves[-1])
                bot_move = chaturanga.bot(Chessboard)[0]
                Chessboard.move(bot_move)
                lichess.bots.make_move(game_id, bot_move)
