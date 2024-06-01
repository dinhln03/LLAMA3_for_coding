import gc
import string
import random


class ActiveGarbageCollection:

    def __init__(self, title):
        assert gc.isenabled(), "Garbage collection should be enabled"
        self.title = title

    def __enter__(self):
        self._collect("start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._collect("completion")

    def _collect(self, step):
        n = gc.collect()
        if n > 0:
            print(f"{self.title}: freed {n} unreachable objects on {step}")


def is_corrupted(entity_max_eventid, last_eventid):
    if last_eventid is None and entity_max_eventid is None:
        # no events, no entities
        return False
    elif last_eventid is not None and entity_max_eventid is None:
        # events but no data (apply has failed or upload has been aborted)
        return False
    elif entity_max_eventid is not None and last_eventid is None:
        # entities but no events (data is corrupted)
        return True
    elif entity_max_eventid is not None and last_eventid is not None:
        # entities and events, entities can never be newer than events
        return entity_max_eventid > last_eventid


def get_event_ids(storage):
    """Get the highest event id from the entities and the eventid of the most recent event

    :param storage: GOB (events + entities)
    :return:highest entity eventid and last eventid
    """
    with storage.get_session():
        entity_max_eventid = storage.get_entity_max_eventid()
        last_eventid = storage.get_last_eventid()
        return entity_max_eventid, last_eventid


def random_string(length):
    """Returns a random string of length :length: consisting of lowercase characters and digits

    :param length:
    :return:
    """
    assert length > 0
    characters = string.ascii_lowercase + ''.join([str(i) for i in range(10)])
    return ''.join([random.choice(characters) for _ in range(length)])
