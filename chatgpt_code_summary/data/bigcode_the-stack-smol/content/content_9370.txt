from flask_restful import Resource

from tasks import add


class HelloResource(Resource):

    def get(self):

        add.delay(3, 5)

        return {"msg": "get ok"}
