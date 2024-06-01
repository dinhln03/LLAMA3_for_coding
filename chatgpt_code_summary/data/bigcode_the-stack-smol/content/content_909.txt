# inclass/mongo_queries.py
import pymongo
import os
from dotenv import load_dotenv
import sqlite3

load_dotenv()

DB_USER = os.getenv("MONGO_USER", default="OOPS")
DB_PASSWORD = os.getenv("MONGO_PASSWORD", default="OOPS")
CLUSTER_NAME = os.getenv("MONGO_CLUSTER_NAME", default="OOPS")
connection_uri = f"mongodb+srv://{DB_USER}:{DB_PASSWORD}@{CLUSTER_NAME}.mongodb.net/test?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE"
print("----------------")
print("URI:", connection_uri)

client = pymongo.MongoClient(connection_uri)
print("----------------")
print("CLIENT:", type(client), client)
# print(dir(client))
# print("DB NAMES:", client.list_database_names()) #> ['admin', 'local']
db = client.ds14_db  # "ds14_db" or whatever you want to call it
# print("----------------")
# print("DB:", type(db), db)

# collection = db.ds14_pokemon_collection # "ds14_collection" or whatever you want to call it
# print("----------------")
# print("COLLECTION:", type(collection), collection)
# print("----------------")
# # print("COLLECTIONS:")
# # print(db.list_collection_names())
# print("--------------------------------------")


################## ASSIGNMENT III #############################

# INSERT RPG DATA INTO MONGODB INSTANCE

# Create RPG database
db = client.rpg_data_db

# Establish sqlite3 connection to access rpg data
sl_conn = sqlite3.connect("data/rpg_db_original.sqlite3")
sl_curs = sl_conn.cursor()


################# CHARACTERS ###########################

# ## Create new collection for RPG data
# col_characters = db.character_collection
# ## Establish SQL syntax for query
# rpg_characters = 'SELECT * FROM charactercreator_character'


# # Function to loop through characters and return list of dictionaries
# def all_chars():
#     query = rpg_characters
#     chars = sl_curs.execute(query)
#     char_data = []
#     for row in chars:
#         character = {
#             "character_id": row[0],
#             "name": row[1],
#             "level": row[2],
#             "exp": row[3],
#             "hp": row[4],
#             "strength": row[5],
#             "intelligence": row[6],
#             "dexterity": row[7],
#             "wisdom": row[8]
#             }
#         char_data.append(character)
#     result = char_data
#     return result

# character_dict_list = all_chars()
# # print(character_dict_list)

# col_characters.insert_many(character_dict_list)
# print("DOCS(Num Characters):", col_characters.count_documents({})) #
# SELECT count(distinct id) from characters


################# MAGES ###########################

# col_mage = db.mage_collection

# mages = 'SELECT * FROM charactercreator_mage'
# def all_chars():
#     query = mages
#     chars = sl_curs.execute(query)
#     char_data = []
#     for row in chars:
#         character = {
#             "character_ptr_id": row[0],
#             "has_pet": row[1],
#             "mana": row[2],
#             }
#         char_data.append(character)
#     result = char_data
#     return result

# character_dict_list = all_chars()

# col_mage.insert_many(character_dict_list)
# print("DOCS:", col_mage.count_documents({}))


################# THIEVES ###########################

# col_thief = db.thief_collection

# thieves = 'SELECT * FROM charactercreator_thief'
# def all_chars():
#     query = thieves
#     chars = sl_curs.execute(query)
#     char_data = []
#     for row in chars:
#         character = {
#             "character_ptr_id": row[0],
#             "is_sneaking": row[1],
#             "energy": row[2],
#             }
#         char_data.append(character)
#     result = char_data
#     return result

# character_dict_list = all_chars()

# col_thief.insert_many(character_dict_list)
# print("DOCS:", col_thief.count_documents({}))


################# CLERICS ###########################

# col_cleric = db.cleric_collection

# clerics = 'SELECT * FROM charactercreator_cleric'
# def all_chars():
#     query = clerics
#     chars = sl_curs.execute(query)
#     char_data = []
#     for row in chars:
#         character = {
#             "character_ptr_id": row[0],
#             "using_shield": row[1],
#             "mana": row[2],
#             }
#         char_data.append(character)
#     result = char_data
#     return result

# character_dict_list = all_chars()

# col_cleric.insert_many(character_dict_list)
# print("DOCS:", col_cleric.count_documents({}))


################# FIGHTERS ###########################

# col_fighter = db.fighter_collection

# fighters = 'SELECT * FROM charactercreator_fighter'
# def all_chars():
#     query = fighters
#     chars = sl_curs.execute(query)
#     char_data = []
#     for row in chars:
#         character = {
#             "character_ptr_id": row[0],
#             "using_shield": row[1],
#             "rage": row[2],
#             }
#         char_data.append(character)
#     result = char_data
#     return result

# character_dict_list = all_chars()

# col_fighter.insert_many(character_dict_list)
# print("DOCS:", col_fighter.count_documents({}))


################# NECROMANCERS ###########################

# col_mancer = db.mancer_collection

# mancers = 'SELECT * FROM charactercreator_necromancer'
# def all_chars():
#     query = mancers
#     chars = sl_curs.execute(query)
#     char_data = []
#     for row in chars:
#         character = {
#             "mage_ptr_id": row[0],
#             "talisman_charged": row[1],
#             }
#         char_data.append(character)
#     result = char_data
#     return result

# character_dict_list = all_chars()

# col_mancer.insert_many(character_dict_list)
# print("DOCS:", col_mancer.count_documents({}))


################# ITEMS ###########################

# col_items = db.items_collection

# items = 'SELECT * FROM armory_item'
# def all_chars():
#     query = items
#     chars = sl_curs.execute(query)
#     char_data = []
#     for row in chars:
#         character = {
#             "item_id": row[0],
#             "name": row[1],
#             "value": row[2],
#             "weight": row[3]
#             }
#         char_data.append(character)
#     result = char_data
#     return result

# character_dict_list = all_chars()

# col_items.insert_many(character_dict_list)
# print("DOCS:", col_items.count_documents({}))


################# WEAPONS ###########################

# col_weapons = db.weapons_collection

# weapons = 'SELECT * FROM armory_weapon'
# def all_chars():
#     query = weapons
#     chars = sl_curs.execute(query)
#     char_data = []
#     for row in chars:
#         character = {
#             "item_ptr_id": row[0],
#             "power": row[1]
#             }
#         char_data.append(character)
#     result = char_data
#     return result

# character_dict_list = all_chars()

# col_weapons.insert_many(character_dict_list)
# print("DOCS:", col_weapons.count_documents({}))


################# INVENTORY ###########################

# col_inventory = db.inventory_collection

# records = 'SELECT * FROM charactercreator_character_inventory'
# def all_chars():
#     query = records
#     chars = sl_curs.execute(query)
#     char_data = []
#     for row in chars:
#         character = {
#             "id": row[0],
#             "character_id": row[1],
#             "item_id": row[2]
#             }
#         char_data.append(character)
#     result = char_data
#     return result

# character_dict_list = all_chars()

# col_inventory.insert_many(character_dict_list)
# print("DOCS:", col_inventory.count_documents({}))
# print("COLLECTIONS:")
# print(db.list_collection_names())


#################### IN-CLASS POKEMON INSERTS #############################


# collection.insert_one({
#     "name": "Pikachu",
#     "level": 30,
#     "exp": 76000000000,
#     "hp": 400,
#     "fav_icecream_flavors":["vanila_bean", "choc"],
#     "stats":{"a":1,"b":2,"c":[1,2,3]}
# })
# print("DOCS:", collection.count_documents({})) # SELECT count(distinct id) from pokemon
# print(collection.count_documents({"name": "Pikachu"})) # SELECT
# count(distinct id) from pokemon WHERE name = "Pikachu"


# mewtwo = {
#     "name": "Mewtwo",
#     "level": 100,
#     "exp": 76000000000,
#     "hp": 450,
#     "strength": 550,
#     "intelligence": 450,
#     "dexterity": 300,
#     "wisdom": 575
# }
# blastoise = {
#     "name": "Blastoise",
#     "lvl": 70, # OOPS we made a mistake with the structure of this dict
# }
# charmander = {
#     "nameeeeeee": "Charmander",
#     "level": 70,
#     "random_stat": {"a":2}
# }
# skarmory = {
#     "name": "Skarmory",
#     "level": 22,
#     "exp": 42000,
#     "hp": 85,
#     "strength": 750,
#     "intelligence": 8,
#     "dexterity": 57
# }
# cubone = {
#     "name": "Cubone",
#     "level": 20,
#     "exp": 35000,
#     "hp": 80,
#     "strength": 600,
#     "intelligence": 60,
#     "dexterity": 200,
#     "wisdom": 200
# }
# scyther = {
#     "name": "Scyther",
#     "level": 99,
#     "exp": 7000,
#     "hp": 40,
#     "strength": 50,
#     "intelligence": 40,
#     "dexterity": 30,
#     "wisdom": 57
# }
# slowpoke = {
#     "name": "Slowpoke",
#     "level": 1,
#     "exp": 100,
#     "hp": 80,
#     "strength": 100,
#     "intelligence": 10,
#     "dexterity": 50,
#     "wisdom": 200
# }


# pokemon_team = [mewtwo, blastoise, skarmory, cubone, scyther, slowpoke, charmander]


# collection.insert_many(pokemon_team)
# print("DOCS:", collection.count_documents({})) # SELECT count(distinct id) from pokemon
# #collection.insert_one({"_id": "OURVAL", "name":"TEST"})
# # can overwrite the _id but not insert duplicate _id values
# #breakpoint()


# pikas = list(collection.find({"name": "Pikachu"})) # SELECT * FROM pokemon WHERE name = "Pikachu"
# # print(len(pikas), "PIKAS")
# # print(pikas[0]["_id"]) #> ObjectId('5ebc31c79c171e43bb5ed469')
# # print(pikas[0]["name"])


# # strong = list(collection.find({"level": {"$gte": 60}} $or {"lvl": {"$gte": 60}}))
# # strong = list(collection.find({"level": {"$gte": 60}, "$or" "lvl": {"$gte": 60}}))
# strong = list(collection.find({"$or": [{"level": {"$gte": 60}}, {"lvl": {"$gte": 60}}]}))
# # TODO: also try to account for our mistakes "lvl" vs "level"
# breakpoint()
# print(strong)
