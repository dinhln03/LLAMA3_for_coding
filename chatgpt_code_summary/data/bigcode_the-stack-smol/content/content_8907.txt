from flask import jsonify
from flask_restful import Resource


class SKU(Resource):

    def get(self):
        return jsonify(
            {}
        )
