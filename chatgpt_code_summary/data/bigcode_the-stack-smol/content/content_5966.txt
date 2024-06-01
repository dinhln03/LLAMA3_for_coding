#%% First
import numpy as np
import json
import os
import pandas as pd
import requests
from contextlib import closing
import time
from datetime import datetime
from requests.models import HTTPBasicAuth
import seaborn as sns
from matplotlib import pyplot as plt
from requests import get
from requests_futures.sessions import FuturesSession
from bs4 import BeautifulSoup

from dotenv import load_dotenv, dotenv_values
from requests_oauthlib import OAuth2, OAuth2Session

#%%
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

env_vars = dotenv_values('config.env')
client_id = env_vars['id']
client_secret = env_vars['secret']
code = env_vars['code']

callback_uri = "http://localhost:8080"
authorize_url = "https://www.warcraftlogs.com/oauth/authorize"
token_url = "https://www.warcraftlogs.com/oauth/token"

# warcraftlogs = OAuth2Session(client_id, redirect_uri=callback_uri)
# authorization_url, state = warcraftlogs.authorization_url(authorize_url,
#         access_type="offline")

# token = warcraftlogs.fetch_token(token_url = token_url,
#                                  auth = HTTPBasicAuth(client_id, client_secret),
#                                  code = code)
# access_token = token['access_token']
# refresh_token = token['refresh_token']
# with open('refresh_token.env', 'w') as f:
#     f.write('refresh_token = '+str(refresh_token)+'\nacces_token = '+str(access_token))

if os.path.isfile('refresh_token.env'):
    env_vars = dotenv_values('refresh_token.env')
    refresh_token = env_vars['refresh_token']
    access_token = env_vars['access_token']
else:
    raise 'Get your fresh token dumby'

# print(refresh_token)
try:
    warcraftlogs = OAuth2Session(client_id = client_id)
    graphql_endpoint = "https://www.warcraftlogs.com/api/v2/client"
    headers = {"Authorization": f"Bearer {access_token}"}

    query = """{
    reportData{
        reports(guildID: 95321, endTime: 1622872800000.0, startTime: 1605855600000.0){
        data{
            fights(difficulty: 5){
            name          
            averageItemLevel
            #   friendlyPlayers
            id
            }
        }
        }
    }
    }"""

    r = requests.post(graphql_endpoint, json={"query": query}, headers=headers)    
except:
    token = warcraftlogs.refresh_token(token_url = token_url,
                                    auth = HTTPBasicAuth(client_id, client_secret),
                                    refresh_token = refresh_token)
    access_token = token['access_token']
    refresh_token = token['refresh_token']
    with open('refresh_token.env', 'w') as f:
        f.write('refresh_token = '+str(refresh_token)+'\naccess_token = '+str(access_token))
        
    warcraftlogs = OAuth2Session(client_id = client_id)
    graphql_endpoint = "https://www.warcraftlogs.com/api/v2/client"
    headers = {"Authorization": f"Bearer {access_token}"}

    query = """{
    reportData{
        reports(guildID: 95321, endTime: 1622872800000.0, startTime: 1605855600000.0){
        data{
            fights(difficulty: 5){
            name          
            averageItemLevel
            #   friendlyPlayers
            id
            }
        }
        }
    }
    }"""

    r = requests.post(graphql_endpoint, json={"query": query}, headers=headers)    

with open('..//get_guild_list/guild_list_hungering.json', encoding='utf-8') as f:
    guilds = json.load(f)

#%%

def is_good_response_json(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('json') > -1)

def get_guild_id(guild):
    try:
        guild_id = int(guild['id'])
    except:
        query = """    
            {
            guildData{
                guild(name: "%s", serverSlug: "%s", serverRegion: "%s"){
                id
                }
            }
            }
        """ % (guild['name'], guild['realm'].replace(' ', '-'), guild['region'])
        r = requests.post(graphql_endpoint, json={"query": query}, headers=headers)
        guild_id = r.json()['data']['guildData']['guild']['id']
    return guild_id

def get_log_list(guild):
    guild['id'] = get_guild_id(guild)
    query = ("{"
    f"reportData{{"
    f"    reports(guildID: {guild['id']}, zoneID: 26){{"
    f"    data{{"
    f"        code"
    f"        startTime"
    f"        endTime"
    f"    }}"
    f"    }}"
    f"}}"
    f"}}")
    r = requests.post(graphql_endpoint, json={"query": query}, headers=headers)
    log_list = r.json()['data']['reportData']['reports']['data']

    return log_list

def get_log_list_apiv1(guild):
    with open('..//..//Warcraftlogs//api_key.txt.') as f:
            api_key = f.readlines()[0]
    
    link = "https://www.warcraftlogs.com:443/v1/reports/guild/" +  \
            guild['name'] + "/" + guild['realm'].replace(' ', '-').replace("'","")+ "/" + \
            guild['region'] + "?api_key=" + api_key

    guild_logs = requests.get(link)
    log_list = guild_logs.json()

    log_list_new = []
    for item in log_list:
        if item['zone'] == 26:
            log_list_new.append({'code': item['id'],
                                'startTime': item['start'],
                                'endTime': item['end']})
                
    return log_list_new

def get_pulls(log, guild):
    log_id = log['code']
    query = """
    {
    reportData{
        report(code: "%s"){
        fights(difficulty: 5){
            name
            id
            averageItemLevel
            bossPercentage
            kill
            startTime
            endTime
        }
        }
    }
    }
    """ % (log_id)

    r = requests.post(graphql_endpoint, json={"query": query}, headers=headers)
    fight_list = r.json()['data']['reportData']['report']['fights']
    for k in range(len(fight_list)):
        fight_list[k].update({'log_code': log_id})    
    return fight_list

def get_fight_info(fight, guild, unique_id):
    code = fight['log_code']
    fight_ID = fight['id']
    start_time = fight['start_time']
    end_time = fight['end_time']
    query = """
    {
    reportData{
        report(code: "%s"){
        table(fightIDs: %s, startTime: %s, endTime: %s)
        }
    }
    }
    """ % (code, fight_ID, str(start_time), str(end_time))
    r = requests.post(graphql_endpoint, json={"query": query}, headers=headers)
    table = r.json()['data']['reportData']['report']['table']['data']
    comp = table['composition']
    roles = table['playerDetails']
    player_list = []
    for role in roles:
        players = roles[role]
        for player in players:
            try:
                gear_ilvl = [piece['itemLevel'] for piece in player['combatantInfo']['gear']]
                ilvl = np.mean(gear_ilvl)
            except:
                try:
                    ilvl = player['minItemLevel']
                except:
                    ilvl = np.NaN

            try:
                covenant = player['combatantInfo']['covenantID']
            except:
                covenant = np.NaN

            try:
                spec = player['specs'][0]
            except:
                spec = np.NaN

            try:
                stats = player['combatantInfo']['stats']
                primaries = ['Agility','Intellect','Strength']
                for primary in primaries:
                    if primary in stats.keys():
                        break
                primary= stats[primary]['min']
                mastery= stats['Mastery']['min']
                crit= stats['Crit']['min']
                haste= stats['Haste']['min']
                vers= stats['Versatility']['min']
                stamina= stats['Stamina']['min']
            except:
                primary = np.NaN
                mastery = np.NaN
                crit = np.NaN
                haste = np.NaN
                vers = np.NaN
                stamina = np.NaN
        
            player_info= {'unique_id': unique_id,
                        'class': player['type'],
                        'spec': spec,
                        'role': role,
                        'ilvl': ilvl,
                        'covenant': covenant,
                        'primary': primary,
                        'mastery': mastery,
                        'crit': crit,
                        'haste': haste,
                        'vers': vers,
                        'stamina': stamina,
                        'boss_name': fight['name']}
            player_list.append(player_info)
    return player_list


# %% Setup the SQL Stuff
from sqlalchemy import create_engine
import psycopg2
server = 'localhost'
database = 'nathria_prog'
username = 'postgres'
password = 'postgres'

if 'conn' in locals():
    conn.close()
engine = create_engine('postgresql://postgres:postgres@localhost:5432/nathria_prog')
conn = psycopg2.connect('host='+server+' dbname='+database+' user='+username+' password='+password)
curs = conn.cursor()

curs.execute("select exists(select * from information_schema.tables where table_name=%s)",\
    ('nathria_prog_v2',))
if curs.fetchone()[0]:
    curs.execute('select distinct guild_name from nathria_prog_v2')
    already_added_guilds = [item[0] for item in curs.fetchall()]
    already_added_length = len(already_added_guilds)
else:
    already_added_guilds = []
    already_added_length = 0

def check_in_sql(fight):
    unique_id = fight['unique_id']
    curs.execute("select * from nathria_prog_v2 where unique_id = '%s'" % (unique_id))
    if curs.fetchone() is None:
        check_one = False
    else:
        check_one = True

    curs.execute("select * from nathria_prog_v2 where start_time > %s and end_time < %s and guild_name = '%s';" \
        % (fight['start_time']-60, fight['end_time']+60, fight['guild_name']))
    if curs.fetchone() is None:
        check_two = False
    else:
        check_two = True
    check = check_one or check_two
    return check

def add_to_sql(curs, table, info):
    placeholders = ', '.join(['%s'] * len(info))
    columns = ', '.join(info.keys())
    sql = "INSERT INTO %s ( %s ) VALUES ( %s )" % (str(table), columns, placeholders)
    curs.execute(sql, list(info.values()))

#%% This is for futures use
    
def make_logs_query(log):
    log_id = log['code']
    query = """
    {
    reportData{
        report(code: "%s"){
        fights(difficulty: 5){
            name
            id
            averageItemLevel
            bossPercentage
            kill
            startTime
            endTime
        }
        }
    }
    }
    """ % (log_id)

    return query

def get_log_args(log, graphql_endpoint, headers):
    args = {'url': graphql_endpoint,
            'json': {'query': make_logs_query(log)},
            'headers': headers}
    return args

def get_fight_list(log_list, graphql_endpoint, headers):
    session = FuturesSession(max_workers = 2)
    futures = [session.post(**get_log_args(log, graphql_endpoint, headers)) for log in log_list]

    fights_list = []
    for q, item in enumerate(futures):
        result = item.result()
        if result.status_code!=200:
            print(result.status_code)
        fights = result.json()['data']['reportData']['report']['fights']
        for k, fight in enumerate(fights):
            fight['log_code'] = log_list[q]['code']
            fight['log_start'] = log_list[q]['startTime']
            fight['log_end'] = log_list[q]['endTime']
            fight['unique_id'] = log_list[q]['code'] + '_' + str(fight['id'])
            fights_list.extend([fight])
    
    return fights_list

def get_prog_pulls(df, boss_name):
    if type(df.iloc[0]['start_time']) != 'int':
        df['start_time'] = [time.mktime(x.to_pydatetime().timetuple()) for x in df['start_time']]
        df['end_time']   = [time.mktime(x.to_pydatetime().timetuple()) for x in df['end_time']]
    kills_df = df.query('name == "'+boss_name+'"').query('zoneDifficulty == 5').query('kill == True')
    first_kill_time = min(kills_df['start_time'])
    return df.query('name == "'+boss_name+'"').query('zoneDifficulty == 5').query('start_time <= '+str(first_kill_time))

def add_pull_num(df):
    df = df.sort_values(by = ['start_time'])
    df.insert(loc = 0, column = 'pull_num', value = np.arange(len(df))+1)
    return df

def combine_boss_df(df):
    boss_names = [
        'Shriekwing', \
        'Huntsman Altimor',
        'Hungering Destroyer', \
        "Sun King's Salvation",
        "Artificer Xy'mox", \
        'Lady Inerva Darkvein', \
        'The Council of Blood', \
        'Sludgefist', \
        'Stone Legion Generals', \
        'Sire Denathrius']
    combine_df = pd.DataFrame()
    for k, boss_name in enumerate(np.unique(df['name'])):
        if boss_name in boss_names and boss_name in np.unique(df['name']):
            combine_df = combine_df.append(add_pull_num(df.copy(deep = True).query('name == "'+boss_name+'"')))
    combine_df = combine_df.reset_index().drop(columns = 'index')
    return combine_df

n_start = 3500
for gnum, guild in enumerate(guilds[n_start:]):
    if guild['name'] in already_added_guilds:
        continue
    # log_list = get_log_list(guild)
    try:
        log_list = get_log_list_apiv1(guild)
        if len(log_list) == 0:
            print(f'Log list empty for {guild["name"]}')
        fightdf = pd.DataFrame()
        playerdf = pd.DataFrame()
        print(f'Parsing guild {guild["name"]} (#{gnum+1+n_start} of {len(guilds)})')
        fight_list = get_fight_list(log_list, graphql_endpoint, headers)
        fightdf = pd.DataFrame()
        for q, fight in enumerate(fight_list):
            fight['boss_perc'] = fight.pop('bossPercentage')
            fight['average_item_level'] = fight.pop('averageItemLevel')
            fight['unique_id'] = fight['log_code'] + '_' + str(fight['id'])
            fight['start_time'] = fight.pop('startTime')
            fight['end_time'] = fight.pop('endTime')
            fight['guild_name'] = guild['name']
            fight['guild_realm'] = guild['realm']
            fight['guild_region'] = guild['region']
            fightdf = fightdf.append(pd.DataFrame(fight, index=['i',]))
        fightdf = combine_boss_df(fightdf.copy(deep = True))
        fightdf.to_sql('nathria_prog_v2', engine, if_exists='append')
        if len(fightdf)>1:
            print(f'Adding to SQL guild {guild["name"]}')
        time.sleep(3)
    except:
        continue

#%%
asdfasdf
from sqlalchemy import create_engine
import psycopg2
server = 'localhost'
database = 'nathria_prog'
username = 'postgres'
password = 'postgres'

if 'conn' in locals():
    conn.close()
engine = create_engine('postgresql://postgres:postgres@localhost:5432/nathria_prog')
conn = psycopg2.connect('host='+server+' dbname='+database+' user='+username+' password='+password)
curs = conn.cursor()

curs.execute("select exists(select * from information_schema.tables where table_name=%s)",\
    ('nathria_prog_v2',))
if curs.fetchone()[0]:
    curs.execute('select distinct guild_name from nathria_prog_v2')
    logged_guilds = [item[0] for item in curs.fetchall()]
else:
    logged_guilds = []
    
def make_fights_query(fight):
    code = fight['log_code']
    fight_ID = fight['id']
    start_time = fight['start_time']
    end_time = fight['end_time']
    query = """
    {
    reportData{
        report(code: "%s"){
        table(fightIDs: %s, startTime: %s, endTime: %s)
        }
    }
    }
    """ % (code, fight_ID, str(start_time), str(end_time))

    return query

def get_fight_args(log, graphql_endpoint, headers):
    args = {'url': graphql_endpoint,
            'json': {'query': make_fights_query(log)},
            'headers': headers}
    return args

def get_fight_table(fights_list, graphql_endpoint, headers):
    session = FuturesSession(max_workers = 2)
    futures = [session.post(**get_fight_args(fight, graphql_endpoint, headers)) for fight in fights_list]

    fights_tables = []
    for k, item in enumerate(futures):
        result = item.result()
        if result.status_code!=200:
            print(result.status_code)
        # if is_good_response_json(item.result()):
        try:
            fights_tables.append(result.json()['data']['reportData']['report']['table']['data'])
        except:
            pass
    return fights_tables

def parse_fight_table(table, boss_name, unique_id, guild_name):

    comp = table['composition']
    roles = table['playerDetails']
    player_list = []
    for role in roles:
        players = roles[role]
        for player in players:
            try:
                gear_ilvl = [piece['itemLevel'] for piece in player['combatantInfo']['gear']]
                ilvl = np.mean(gear_ilvl)
            except:
                try:
                    ilvl = player['minItemLevel']
                except:
                    ilvl = np.NaN

            try:
                covenant = player['combatantInfo']['covenantID']
            except:
                covenant = np.NaN

            try:
                spec = player['specs'][0]
            except:
                spec = np.NaN

            try:
                stats = player['combatantInfo']['stats']
                primaries = ['Agility','Intellect','Strength']
                for primary in primaries:
                    if primary in stats.keys():
                        break
                primary= stats[primary]['min']
                mastery= stats['Mastery']['min']
                crit= stats['Crit']['min']
                haste= stats['Haste']['min']
                vers= stats['Versatility']['min']
                stamina= stats['Stamina']['min']
            except:
                primary = np.NaN
                mastery = np.NaN
                crit = np.NaN
                haste = np.NaN
                vers = np.NaN
                stamina = np.NaN
        
            player_info= {'unique_id': unique_id,
                        'name': player['name'],
                        'guild_name': guild_name,
                        'server': player['server'],
                        'class': player['type'],
                        'spec': spec,
                        'role': role,
                        'ilvl': ilvl,
                        'covenant': covenant,
                        'primary': primary,
                        'mastery': mastery,
                        'crit': crit,
                        'haste': haste,
                        'vers': vers,
                        'stamina': stamina,
                        'boss_name': boss_name}
            player_list.append(player_info)
    return player_list

for guild_name in logged_guilds:
    curs.execute(f"select * from nathria_prog_v2 where guild_name = '{guild_name}'")
    pulls = pd.DataFrame(curs.fetchall())
    pulls.columns = [desc[0] for desc in curs.description]
    fights_list = pulls.to_dict('records')

    curs.execute(f"select distinct unique_id from nathria_prog_v2_players where guild_name = '{guild_name}'")
    added_fights = [item[0] for item in curs.fetchall()]
    fight_list = [fight for fight in fights_list if fight['unique_id'] not in added_fights]
    
    if len(fight_list)>1:
        fights_tables = get_fight_table(fights_list, graphql_endpoint, headers)

        playerdf = pd.DataFrame()
        for q, table in enumerate(fights_tables):
            unique_id = fights_list[q]['unique_id']
            guild_name = guild_name
            player_info = parse_fight_table(table, fights_list[q]['name'], unique_id, guild_name)
            for player in player_info:
                for player in player_info:
                    playerdf = playerdf.append(pd.DataFrame(player, index=['i',]))
        if len(playerdf)>1:
            print(f'Adding to SQL guild player info {guild["name"]}')
        playerdf.to_sql('nathria_prog_v2_players', engine, if_exists='append')