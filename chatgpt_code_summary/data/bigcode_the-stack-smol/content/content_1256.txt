#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tornado.gen
import bcrypt

__all__ = ["create_new_user"]


@tornado.gen.coroutine
def get_next_id(db, collection):
    counter = yield db.counters.find_and_modify(
        {"_id": "{}id".format(collection)},
        {"$inc": {"seq": 1}},
        new=True,
    )

    raise tornado.gen.Return(counter["seq"])


@tornado.gen.coroutine
def create_new_user(db, email, password, group):
    password = bcrypt.hashpw(password.encode(), bcrypt.gensalt(8))
    id = yield get_next_id(db, "user")
    yield db.users.insert({
        "_id": id, "email": email, "hash": password, "group": group})
