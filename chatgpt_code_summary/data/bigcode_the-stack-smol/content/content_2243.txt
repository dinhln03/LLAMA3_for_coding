import json
import pickle
from TwitterAPI import TwitterAPI

with open("api_key.json") as json_data:
    all_keys = json.load(json_data)
    consumer_key = all_keys["consumer_key"]
    consumer_secret = all_keys["consumer_secret"]
    access_token_key = all_keys["access_token_key"]
    access_token_secret = all_keys["access_token_secret"]

api = TwitterAPI(consumer_key, consumer_secret, access_token_key, access_token_secret)


master_ID = "116568685"
count = 25

def who_follows(ID):
    page_cursor = get_pickle()
    r = api.request("followers/ids", {"user_id":ID, "cursor":page_cursor, "count":count})
    print(r.status_code)
    parse_response = r.json()
    users_inf = parse_response["ids"]
    IDS = []
    for x in users_inf:
        IDS.append(x)
    page_cursor += -1
    print(page_cursor)
    make_pickle(page_cursor)
    print(IDS)
    return IDS


def make_pickle(obj):
    with open("objs.pkl", "wb") as f:
        pickle.dump(obj, f)

def get_pickle():
    with open("objs.pkl", "rb") as f:
        obj = pickle.load(f)
        print(obj)
        return obj
