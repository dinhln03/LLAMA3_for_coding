from flask import jsonify, request
from flask_restx import Resource, reqparse, fields, marshal_with
import requests
import redis

import os
import logging
import time
import datetime
import json

from app import api, db
from models import User

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

user_fields = {
    "id": fields.Integer,
    "uuid": fields.Integer,
    "status": fields.String
}


@api.route("/users")
class Users(Resource):

    users_post_reqparser = reqparse.RequestParser()
    users_post_reqparser.add_argument(
        "uuid",
        type=int,
        location="json",
        required=True,
        help="Please provide the UUID -",
    )

    @api.expect(users_post_reqparser)
    @marshal_with(user_fields)
    def post(self):
        args = self.users_post_reqparser.parse_args()
        new_user = User(uuid=args["uuid"])
        db.session.add(new_user)
        db.session.flush()
        db.session.commit()
        return new_user, 201

    @marshal_with(user_fields)
    def get(self):
        # TODO: some authorization would be nice
        return User.query.all(), 200


@api.route("/usersByUUID/<int:uuid>")
class UserByUUID(Resource):
    
    @marshal_with(user_fields)
    def get(self, uuid):
        user = User.query.filter_by(uuid=uuid).first()
        if user is None:
            # we should really return 404 here and don't do POST magic 
            # in a GET request but this will make some thing much easier...
            user = User(uuid=uuid)
            db.session.add(user)
            db.session.flush()
            db.session.commit()
        return user, 200


@api.route("/users/<int:id>")
class SingleUser(Resource):

    user_put_reqparser = reqparse.RequestParser()
    user_put_reqparser.add_argument(
        "status",
        type=str,
        location="json",
        required=True,
        help="Please provide the status value (healty, covid_positive, covid_negative) -",
    )

    @marshal_with(user_fields)
    def get(self, id):
        found_user = User.query.filter_by(uuid=id).first()
        if found_user is None:
            api.abort(404, "User does not exist.")
        return found_user, 200

    @marshal_with(user_fields)
    def put(self, id):
        user = User.query.filter_by(uuid=id).first()
        if user is None:
            api.abort(404, "User does not exist.")

        args = self.user_put_reqparser.parse_args()
        user.status = args["status"]
        db.session.commit()
        if args["status"] == "covid_positive":
            self._submit_filtering_jobs(user.uuid)

        return user, 200

    def delete(self, id):
        user = User.query.filter_by(uuid=id).first()
        if user is None:
            api.abort(404, "User does not exist.")

        db.session.delete(user)
        db.session.commit()
        return {"msg": "ok"}, 200

    @staticmethod
    def _chunks(l, n):
        n = max(1, n)
        return (l[i : i + n] for i in range(0, len(l), n))

    def _submit_filtering_jobs(self, uuid):
        """
        Here we create the task and put it on the job queue.
        """
        # Some optimization: we make a request to the Location API
        # to get all the geohash prefixes for all locations the diagonzed patient
        # has visited in the last two weeks
        two_weeks_ago = datetime.date.today() - datetime.timedelta(14)
        params = {
            "from": int(two_weeks_ago.strftime("%s")),
            "to": int(time.time()),
            "unit": "seconds",
        }
        # TODO: Do not hardcode URIs or ports, use env vars instead
        # TODO: Do not assume that the period is always 2 weeks long, make it parametrized
        location_api_resp = requests.get(
            f"http://location-api:5000/geohashRegionsForUser/{uuid}", params=params
        )
        if location_api_resp.status_code != 200:
            logger.warning(location_api_resp)
            api.abort(
                500, "There was a problem when requesting data from the Location API"
            )
        visited_regions_geohash_prefixes = location_api_resp.json()
        logger.info(f"Visited Regions for diagonzed patient: {str(visited_regions_geohash_prefixes)}")

        location_api_resp_users = requests.get("http://location-api:5000/users")
        if location_api_resp_users.status_code != 200:
            logger.warning(location_api_resp_users)
            api.abort(
                500, "There was a problem when requesting data from the Location API"
            )
        all_influx_users = list(set(location_api_resp_users.json()) - {str(uuid)})
        logger.info(f"All Influx users without diagnozed patient: {str(all_influx_users)}")

        # So, we should split the whole job into rougly N*k jobs, where N is the
        # number of workers listening on the queue, so that each worker will get roughly
        # k tasks to execute (so we can achieve nice load balancing).
        # Let's assume for simplicity now that we have always 3 workers and k = 1.
        n_workers = 3
        task_size = len(all_influx_users) // n_workers
        all_influx_users_partitioned = SingleUser._chunks(all_influx_users, task_size)

        # Create the tasks and put the onto the Redis queue
        redis_instance = redis.Redis(
            host=os.getenv("REDIS_HOST", "queue"),
            port=os.getenv("REDIS_PORT", 6379),
            db=os.getenv("REDIS_DB_ID", 0),
        )
        redis_namespace = os.getenv("REDIS_NAMESPACE", "worker")
        redis_collection = os.getenv("REDIS_COLLECTION", "jobs")
        logger.info(f"Connected with Redis ({redis_namespace}:{redis_collection})")
        for idx, users_batch in enumerate(all_influx_users_partitioned):
            job = {
                "type": "scan_users_locations",
                "args": {
                    "user_id_range": users_batch,
                    "diagnozed_uuid": uuid,
                    "diagnozed_visited_regions": visited_regions_geohash_prefixes,
                },
            }
            redis_instance.rpush(
                f"{redis_namespace}:{redis_collection}", json.dumps(job)
            )
            logger.info(
                f"Successfully pushed job #{idx} to the Job Queue:\n{json.dumps(job)}"
            )

        logger.info("Finished pushing jobs to the Queue.")
 