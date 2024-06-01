from flask import Flask, request, jsonify, send_from_directory
import engine

app = Flask(__name__)

@app.route('/api/texts')
def texts():
    return send_from_directory('i18n', 'ui.de.json');

@app.route('/api/codenames')
def codenames():
    return jsonify(engine.codenames())

@app.route('/api/ready')
def ready():
    return jsonify(engine.ready())

@app.route('/api/clue', methods=['POST'])
def clue():
    content = request.json
    return jsonify(engine.clue(
        our_agents=content['ourAgents'],
        assassin=content['assassin'],
        previous_clues=content['previousClues'],
        min_related=content['minRelated'],
        max_related=content['maxRelated']
        ))

@app.route('/api/guess', methods=['POST'])
def guess():
    content = request.json
    return jsonify(engine.guess(
        codenames=content['codenames'],
        word=content['word'],
        number=content['number']
        ))
