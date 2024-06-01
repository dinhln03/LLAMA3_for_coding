from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime, timedelta
from elasticsearch6 import Elasticsearch
from json import dump, load
from math import pi, sin, cos
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker as mtick
from requests import get
from tweepy import OAuthHandler, API
import traceback


# Multi-day, use gte
battles_query = {
    "aggs": {
        "2": {
            "date_histogram": {
                "field": "date",
                "interval": "1d",
                "min_doc_count": 0
            },
            "aggs": {
                "3": {
                    "terms": {
                        "field": "console.keyword",
                        "size": 2,
                        "order": {
                            "1": "desc"
                        },
                        "min_doc_count": 0
                    },
                    "aggs": {
                        "1": {
                            "sum": {
                                "field": "battles"
                            }
                        }
                    }
                }
            }
        }
    },
    "size": 0,
    "_source": {"excludes": []},
    "stored_fields": ["*"],
    "script_fields": {},
    "docvalue_fields": [
        {
            "field": "date",
            "format": "date_time"
        }
    ],
    "query": {
        "bool": {
            "must": [
                {"match_all": {}},
                {"match_all": {}},
                {
                    "range": {
                        "date": {
                            "gte": None,
                            "lte": None,
                            "format": "date"
                        }
                    }
                }
            ],
            "filter": [],
            "should": [],
            "must_not": []
        }
    }
}

# Multi-day, use gte
players_query = {
    "aggs": {
        "2": {
            "date_histogram": {
                "field": "date",
                "interval": "1d",
                "min_doc_count": 0
            },
            "aggs": {
                "3": {
                    "terms": {
                        "field": "console.keyword",
                        "size": 2,
                        "order": {
                            "_count": "desc"
                        },
                        "min_doc_count": 0
                    }
                }
            }
        }
    },
    "size": 0,
    "_source": {"excludes": []},
    "stored_fields": ["*"],
    "script_fields": {},
    "docvalue_fields": [
        {
            "field": "date",
            "format": "date_time"
        }
    ],
    "query": {
        "bool": {
            "must": [
                {"match_all": {}},
                {"match_all": {}},
                {
                    "range": {
                        "date": {
                            "gte": None,
                            "lte": None,
                            "format": "date"
                        }
                    }
                }
            ],
            "filter": [],
            "should": [],
            "must_not": []
        }
    }
}

unique_count_query = {
    "aggs": {
        "2": {
            "terms": {
                "field": "console.keyword",
                "size": 2,
                "order": {
                    "1": "desc"
                }
            },
            "aggs": {
                "1": {
                    "cardinality": {
                        "field": "account_id"
                    }
                }
            }
        }
    },
    "size": 0,
    "_source": {"excludes": []},
    "stored_fields": ["*"],
    "script_fields": {},
    "docvalue_fields": [
        {
            "field": "date",
            "format": "date_time"
        }
    ],
    "query": {
        "bool": {
            "must": [
                {"match_all": {}},
                {"match_all": {}},
                {
                    "range": {
                        "date": {
                            "gte": None,
                            "lte": None,
                            "format": "date"
                        }
                    }
                }
            ],
            "filter": [],
            "should": [],
            "must_not": []
        }
    }
}

new_players_query = {
    "aggs": {
        "2": {
            "date_histogram": {
                "field": "created_at",
                "interval": "1d",
                "min_doc_count": 0
            },
            "aggs": {
                "3": {
                    "terms": {
                        "field": "console.keyword",
                        "size": 2,
                        "order": {
                            "_count": "desc"
                        },
                        "min_doc_count": 0
                    }
                }
            }
        }
    },
    "size": 0,
    "_source": {"excludes": []},
    "stored_fields": ["*"],
    "script_fields": {},
    "docvalue_fields": [
        {
            "field": "created_at",
            "format": "date_time"
        }
    ],
    "query": {
        "bool": {
            "must": [
                {"match_all": {}},
                {"match_all": {}},
                {
                    "range": {
                        "created_at": {
                            "gte": None,
                            "lt": None,
                            "format": "date"
                        }
                    }
                }
            ],
            "filter": [],
            "should": [],
            "must_not": []
        }
    }
}

personal_players_query = {
    'sort': [],
    '_source': {'excludes': []},
    'aggs': {
        '2': {
            'date_histogram': {
                'field': 'date',
                'interval': '1d',
                'min_doc_count': 0
            }
        }
    },
    'stored_fields': ['_source'],
    'script_fields': {},
    'docvalue_fields': [{'field': 'date', 'format': 'date_time'}],
    'query': {
        'bool': {
            'must': [
                {'match_all': {}},
                {
                    'range': {
                        'date': {
                            'gt': None,
                            'lte': None,
                            'format': 'date'
                        }
                    }
                }
            ],
           'filter': [],
           'should': [],
           'must_not': []
        }
    },
    'size': 500
}

accounts_per_battles_range_query = {
    'aggs': {
        '2': {
            'range': {
                'field': 'battles',
                'ranges': [
                    {'from': 1, 'to': 5},
                    {'from': 5, 'to': 10},
                    {'from': 10, 'to': 20},
                    {'from': 20, 'to': 30},
                    {'from': 30, 'to': 40},
                    {'from': 40, 'to': 50},
                    {'from': 50}
                ],
                'keyed': True
            },
            'aggs': {
                '3': {
                    'terms': {
                        'field': 'console.keyword',
                        'size': 2,
                        'order': {'_count': 'desc'}
                    }
                }
            }
        }
    },
    'size': 0,
    '_source': {'excludes': []},
    'stored_fields': ['*'],
    'script_fields': {},
    'docvalue_fields': [{'field': 'date', 'format': 'date_time'}],
    'query': {
        'bool': {
            'must': [
                {'match_all': {}},
                {'match_all': {}},
                {'range': {'date': {'gt': None, 'lte': None, 'format': 'date'}}}
            ],
            'filter': [],
            'should': [],
            'must_not': []
        }
    }
}

five_battles_a_day_query = {
    'aggs': {
        '4': {
            'date_histogram': {
                'field': 'date',
                'interval': '1d',
                'min_doc_count': 0
            },
            'aggs': {
                '3': {
                    'terms': {
                        'field': 'console.keyword',
                        'size': 2,
                        'order': {'_count': 'desc'}
                    },
                    'aggs': {
                        '2': {
                            'range': {
                                'field': 'battles',
                                'ranges': [{'from': 5, 'to': None}],
                                'keyed': True
                            }
                        }
                    }
                }
            }
        }
    },
    'size': 0,
    '_source': {'excludes': []},
    'stored_fields': ['*'],
    'script_fields': {},
    'docvalue_fields': [{'field': 'date', 'format': 'date_time'}],
    'query': {
        'bool': {
            'must': [
                {'match_all': {}},
                {'match_all': {}},
                {
                    'range': {
                        'date': {
                            'gte': None,
                            'lte': None,
                            'format': 'date'
                        }
                    }
                }
            ],
            'filter': [],
            'should': [],
            'must_not': []
        }
    }
}

CW_TANKS = 'ASSIGN `build_cw_tanks_list(config)` TO ME'

cw_popular_tanks_query = {
    "aggs": {
        "2": {
            "date_histogram": {
                "field": "date",
                "interval": "1d",
                "min_doc_count": 0
            },
            "aggs": {
                "4": {
                    "terms": {
                        "field": "console.keyword",
                        "size": 5,
                        "order": {
                            "1": "desc"
                        }
                    },
                    "aggs": {
                        "1": {
                            "sum": {
                                "field": "battles"
                            }
                        },
                        "3": {
                            "terms": {
                                "field": "tank_id",
                                "size": 5,
                                "order": {
                                    "1": "desc"
                                }
                            },
                            "aggs": {
                                "1": {
                                    "sum": {
                                        "field": "battles"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "size": 0,
    "_source": {
        "excludes": []
    },
    "stored_fields": [
        "*"
    ],
    "script_fields": {},
    "docvalue_fields": [
        {
            "field": "date",
            "format": "date_time"
        }
    ],
    "query": {
        "bool": {
            "must": [
                {
                    "query_string": {
                        "query": CW_TANKS,
                        "analyze_wildcard": True,
                        "default_field": "*"
                    }
                },
                {
                    "range": {
                        "date": {
                            "gte": None,
                            "lte": None,
                            "format": "date"
                        }
                    }
                }
            ],
            "filter": [],
            "should": [],
            "must_not": []
        }
    }
}

ww2_popular_tanks_query = {
    "aggs": {
        "2": {
            "date_histogram": {
                "field": "date",
                "interval": "30m",
                "time_zone": "America/Chicago",
                "min_doc_count": 0
            },
            "aggs": {
                "4": {
                    "terms": {
                        "field": "console.keyword",
                        "size": 5,
                        "order": {
                            "1": "desc"
                        }
                    },
                    "aggs": {
                        "1": {
                            "sum": {
                                "field": "battles"
                            }
                        },
                        "3": {
                            "terms": {
                                "field": "tank_id",
                                "size": 5,
                                "order": {
                                    "1": "desc"
                                }
                            },
                            "aggs": {
                                "1": {
                                    "sum": {
                                        "field": "battles"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "size": 0,
    "_source": {
        "excludes": []
    },
    "stored_fields": [
        "*"
    ],
    "script_fields": {},
    "docvalue_fields": [
        {
            "field": "date",
            "format": "date_time"
        }
    ],
    "query": {
        "bool": {
            "must": [
                {
                    "query_string": {
                        "query": 'NOT (' + CW_TANKS + ')',
                        "analyze_wildcard": True,
                        "default_field": "*"
                    }
                },
                {
                    "range": {
                        "date": {
                            "gte": None,
                            "lte": None,
                            "format": "date"
                        }
                    }
                }
            ],
            "filter": [],
            "should": [],
            "must_not": []
        }
    }
}

BATTLES_PNG = '/tmp/battles.png'
PLAYERS_PNG = '/tmp/players.png'
NEWPLAYERS_PNG = '/tmp/newplayers.png'
AVERAGE_PNG = '/tmp/average.png'
ACCOUNTAGE_PNG = '/tmp/accountage.png'
BATTLERANGE_PNG = '/tmp/battlerange.png'
FIVEADAY_PNG = '/tmp/fiveaday.png'
PLAYERSLONG_PNG = '/tmp/playerslong.png'
BATTLESLONG_PNG = '/tmp/battleslong.png'
AVERAGELONG_PNG = '/tmp/averagelong.png'
MODEBREAKDOWN_PNG = '/tmp/modebreakdown.png'
MODEBREAKDOWNLONG_PNG = '/tmp/modebreakdownlong.png'
MODEBREAKDOWNPERCENT_PNG = '/tmp/modebreakdownpercent.png'
MODEBREAKDOWNPERCENTLONG_PNG = '/tmp/modebreakdownpercentlong.png'

def manage_config(mode, filename='config.json'):
    if mode == 'read':
        with open(filename) as f:
            return load(f)
    elif mode == 'create':
        with open(filename, 'w') as f:
            dump(
                {
                    'days': 14,
                    'long term': 90,
                    'omit errors long term': True,
                    'twitter': {
                        'api key': '',
                        'api secret key': '',
                        'access token': '',
                        'access token secret': '',
                        'message': "Today's update on the active player count and total battles per platform for #worldoftanksconsole."
                    },
                    'elasticsearch': {
                        'hosts': ['127.0.0.1']
                    },
                    'battle index': 'diff_battles-*',
                    'tank index': 'diff_tanks-*',
                    'unique': [7, 14, 30],
                    'account age': [7, 30, 90, 180, 365, 730, 1095, 1460, 1825],
                    'battle ranges': [
                        {"from": 1, "to": 5},
                        {"from": 5, "to": 10},
                        {"from": 10, "to": 20},
                        {"from": 20, "to": 30},
                        {"from": 30, "to": 40},
                        {"from": 40, "to": 50},
                        {"from": 50}
                    ],
                    'watermark text': '@WOTC_Tracker',
                    'wg api key': 'DEMO'
                }
            )


def query_es_for_graphs(config):
    now = datetime.utcnow()
    then = now - timedelta(days=config['days'])
    es = Elasticsearch(**config['elasticsearch'])
    # Setup queries
    battles_query['query']['bool'][
        'must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    battles_query['query']['bool'][
        'must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    players_query['query']['bool'][
        'must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    players_query['query']['bool'][
        'must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    new_players_query['query']['bool'][
        'must'][-1]['range']['created_at']['gte'] = then.strftime('%Y-%m-%d')
    new_players_query['query']['bool'][
        'must'][-1]['range']['created_at']['lt'] = now.strftime('%Y-%m-%d')
    # Query Elasticsearch
    battles = es.search(index=config['battle index'], body=battles_query)
    players = es.search(index=config['battle index'], body=players_query)
    newplayers = es.search(index='players', body=new_players_query)
    # Filter numbers
    battles_xbox = []
    battles_ps = []
    players_xbox = []
    players_ps = []
    newplayers_xbox = []
    newplayers_ps = []
    averages_xbox = []
    averages_ps = []
    for bucket in battles['aggregations']['2']['buckets']:
        if not bucket['3']['buckets']:
            battles_xbox.append(0)
            battles_ps.append(0)
            continue
        for subbucket in bucket['3']['buckets']:
            if subbucket['key'] == 'xbox':
                battles_xbox.append(subbucket['1']['value'])
            else:
                battles_ps.append(subbucket['1']['value'])
    for bucket in players['aggregations']['2']['buckets']:
        if not bucket['3']['buckets']:
            players_xbox.append(0)
            players_ps.append(0)
            continue
        for subbucket in bucket['3']['buckets']:
            if subbucket['key'] == 'xbox':
                players_xbox.append(subbucket['doc_count'])
            else:
                players_ps.append(subbucket['doc_count'])
    for bucket in newplayers['aggregations']['2']['buckets']:
        if not bucket['3']['buckets']:
            newplayers_xbox.append(0)
            newplayers_ps.append(0)
        for subbucket in bucket['3']['buckets']:
            if subbucket['key'] == 'xbox':
                newplayers_xbox.append(subbucket['doc_count'])
            else:
                newplayers_ps.append(subbucket['doc_count'])
    for b, p in zip(battles_xbox, players_xbox):
        averages_xbox.append(b / p)
    for b, p in zip(battles_ps, players_ps):
        averages_ps.append(b / p)
    dates = [b['key_as_string'].split('T')[0] for b in players[
        'aggregations']['2']['buckets']]
    newplayers_dates = [b['key_as_string'].split('T')[0] for b in newplayers[
        'aggregations']['2']['buckets']]
    return dates, battles_xbox, battles_ps, players_xbox, players_ps, newplayers_dates, newplayers_xbox, newplayers_ps, averages_xbox, averages_ps


def query_es_for_unique(config):
    now = datetime.utcnow()
    es = Elasticsearch(**config['elasticsearch'])
    unique = {'Xbox': [], 'Playstation': []}
    unique_count_query['query']['bool'][
        'must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    for earliest in config['unique']:
        unique_count_query['query']['bool']['must'][-1]['range']['date'][
            'gte'] = (now - timedelta(days=earliest)).strftime('%Y-%m-%d')
        results = es.search(index=config['battle index'], body=unique_count_query)
        for bucket in results['aggregations']['2']['buckets']:
            if bucket['key'] == 'xbox':
                unique['Xbox'].append(bucket['1']['value'])
            else:
                unique['Playstation'].append(bucket['1']['value'])
    return unique


def create_activity_graphs(dates, battles_xbox, battles_ps, players_xbox, players_ps, newplayers_dates, newplayers_xbox, newplayers_ps, averages_xbox, averages_ps, watermark_text='@WOTC_Tracker'):
    shifted_dates = [(datetime.strptime(d, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d') for d in dates]
    # Players PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle('Active Accounts Per Platform')
    # ax1 = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticklabels(shifted_dates, ha='right')
    ax1.plot(shifted_dates, players_xbox, color='green', linewidth=2, label='Xbox')
    ax1.plot(shifted_dates, players_ps, color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(PLAYERS_PNG)
    del fig
    # Battles PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle('Total Battles Per Platform')
    # ax = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticklabels(shifted_dates, ha='right')
    ax1.plot(shifted_dates, battles_xbox, color='green', linewidth=2, label='Xbox')
    ax1.plot(shifted_dates, battles_ps, color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(BATTLES_PNG)
    del fig
    # New Players PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle('New Accounts Per Platform')
    # ax = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticklabels(dates, ha='right')
    ax1.plot(newplayers_dates, newplayers_xbox, color='green', linewidth=2, label='Xbox')
    ax1.plot(newplayers_dates, newplayers_ps, color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(NEWPLAYERS_PNG)
    del fig
    # Averages PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle('Average Battles Played Per Account Per Platform')
    # ax = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticklabels(shifted_dates, ha='right')
    ax1.plot(shifted_dates, averages_xbox, color='green', linewidth=2, label='Xbox')
    ax1.plot(shifted_dates, averages_ps, color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(AVERAGE_PNG)
    del fig


def query_es_for_active_accounts(config):
    now = datetime.utcnow()
    then = now - timedelta(days=1)
    es = Elasticsearch(**config['elasticsearch'])
    personal_players_query['query']['bool']['must'][-1]['range']['date']['gt'] = then.strftime('%Y-%m-%d')
    personal_players_query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')

    # Get all account IDs of active players
    hits = []
    response = es.search(index=config['battle index'], body=personal_players_query, scroll='30s')
    while len(response['hits']['hits']):
        hits.extend(response['hits']['hits'])
        response = es.scroll(scroll_id=response['_scroll_id'], scroll='3s')

    flattened = [doc['_source']['account_id'] for doc in hits]

    # Query account information to get age details
    player_info_extracted = []
    for i in range(0, len(flattened), 10000):
        active_player_info = es.mget(index='players', doc_type='player', body={'ids': flattened[i:i+10000]}, _source=['account_id', 'console', 'created_at'])
        player_info_extracted.extend([doc['_source'] for doc in active_player_info['docs']])

    sorted_player_info = sorted(player_info_extracted, key = lambda d: d['created_at'])
    buckets = {
        "xbox": OrderedDict((v, 0) for v in sorted(config['account age'])),
        "ps": OrderedDict((v, 0) for v in sorted(config['account age'])),
        "all": OrderedDict((v, 0) for v in sorted(config['account age']))
    }

    # Sum account ages based on range of age
    buckets['xbox']['other'] = 0
    buckets['ps']['other'] = 0
    buckets['all']['other'] = 0
    for player in sorted_player_info:
        delta = now - datetime.strptime(player['created_at'], '%Y-%m-%dT%H:%M:%S')
        for key in buckets['all'].keys():
            if not isinstance(key, int):
                buckets['all'][key] += 1
                buckets[player['console']][key] += 1
                break
            elif delta.total_seconds() <= (key * 24 * 60 * 60):
                buckets['all'][key] += 1
                buckets[player['console']][key] += 1
                break
    return buckets


def calc_label(value):
    if value < 7:
        return '{} day{}'.format(value, '' if value == 1 else 's')
    elif 7 <= value < 30:
        return '{} week{}'.format(value // 7, '' if value // 7 == 1 else 's')
    elif 30 <= value < 365:
        return '{} month{}'.format(value // 30, '' if value // 30 == 1 else 's')
    else:
        return '{} year{}'.format(value // 365, '' if value // 365 == 1 else 's')


def calc_angle(wedge):
    return (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1


def create_account_age_chart(buckets, watermark_text='@WOTC_Tracker'):
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    then = datetime.utcnow() - timedelta(days=1)
    fig.suptitle("Breakdown of active accounts by account age for {}".format(then.strftime('%Y-%m-%d')))
    ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=10)
    ax1.axis('equal')
    size = 0.125

    outer_labels = []
    prev = 0
    for key in buckets['all'].keys():
        if not isinstance(key, int):
            outer_labels.append('>' + calc_label(prev))
        else:
            outer_labels.append('{} - {}'.format(calc_label(prev), calc_label(key)))
            prev = key

    # Outer pie chart
    outer_cmap = plt.get_cmap("binary")
    outer_colors = outer_cmap([i * 10 for i in range(10, len(buckets['all'].keys()) + 11)])
    outer_wedges, outer_text, outer_autotext = ax1.pie(
        buckets['all'].values(),
        explode=[0.1 for __ in outer_labels],
        radius=1,
        colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor='w'),
        autopct='%1.1f%%',
        pctdistance=1.1
        #labels=outer_labels
    )

    bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle='-'), bbox=bbox_props, zorder=0, va='center')
    for i, wedge in enumerate(outer_wedges):
        angle = calc_angle(wedge)
        y = sin(angle * (pi / 180))
        x = cos(angle * (pi / 180))
        align = 'right' if x < 0 else 'left'
        connectionstyle = 'angle,angleA=0,angleB={}'.format(angle)
        kw['arrowprops'].update({'connectionstyle': connectionstyle})
        ax1.annotate(
            outer_labels[i],
            xy=(x, y),
            xytext=(1.35*(-1 if x < 0 else 1), 1.4*y),
            horizontalalignment=align,
            **kw
        )

    # Inner pie chart
    inner_cmap = plt.get_cmap("tab20c")
    pie_flat = list(zip(buckets['xbox'].values(), buckets['ps'].values()))
    inner_labels = []
    for pair in pie_flat:
        inner_labels.extend(['xbox', 'ps'])
    inner_colors = inner_cmap([1 if console == 'ps' else 9 for console in inner_labels])
    inner_wedges, inner_text, inner_autotext = ax1.pie(
        [item for sublist in pie_flat for item in sublist],
        explode=[0.1 for __ in inner_labels],
        radius=1.05-size,
        colors=inner_colors,
        wedgeprops=dict(width=size, edgecolor='w'),
        autopct='',
        pctdistance=0.9
    )

    # Replace inner text with actual values
    for i, label, wedge, text in zip(range(len(inner_wedges)), inner_labels, inner_wedges, inner_autotext):
        text.set_text(buckets[label]['other' if i // 2 > len(buckets['all'].keys()) - 1 else list(buckets['all'].keys())[i // 2]])
        angle = calc_angle(wedge)
        if 90 < angle < 270:
            angle += 180
        text.set_rotation(angle)

    # Patch inner wedges to group together in explosion
    # Influenced by: https://stackoverflow.com/a/20556088/1993468
    groups = [[i, i+1] for i in range(0, len(inner_wedges), 2)]
    radfraction = 0.1
    for group in groups:
        angle = ((inner_wedges[group[-1]].theta2 + inner_wedges[group[0]].theta1)/2) * (pi / 180)
        for g in group:
            wedge = inner_wedges[g]
            wedge.set_center((radfraction * wedge.r * cos(angle), radfraction * wedge.r * sin(angle)))

    # Add subplot in second row, below nested pie chart
    ax2 = plt.subplot2grid((11, 1), (10, 0))
    ax2.axhline(color='black', y=0)
    # Xbox, Playstation
    totals = [sum(buckets['xbox'].values()), sum(buckets['ps'].values()), sum(buckets['all'].values())]
    ypos = -0.18
    bottom = 0
    height = 0.1
    for i in range(len(totals) - 1):
        width = totals[i] / totals[-1]
        ax2.barh(ypos, width, height, left=bottom, color=inner_colors[i])
        xpos = bottom + ax2.patches[i].get_width() / 2
        bottom += width
        ax2.text(xpos, ypos, '{} ({:.1f}%)'.format(totals[i], (totals[i] / totals[-1]) * 100), ha='center', va='center')

    ax2.axis('off')
    ax2.set_title('Total Active Players', y=0.325)
    ax2.set_xlim(0, 1)

    ax1.legend(inner_wedges[-2:], ['xbox', 'ps'], loc='lower right')
    fig.text(0.5, 0.5, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(ACCOUNTAGE_PNG)
    del fig


def query_es_for_accounts_by_battles(config):
    now = datetime.utcnow()
    then = now - timedelta(days=1)
    es = Elasticsearch(**config['elasticsearch'])
    accounts_per_battles_range_query['query']['bool']['must'][-1]['range']['date']['gt'] = then.strftime('%Y-%m-%d')
    accounts_per_battles_range_query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    if 'battle ranges' in config:
        accounts_per_battles_range_query['aggs']['2']['range']['ranges'] = config['battle ranges']

    response = es.search(index=config['battle index'], body=accounts_per_battles_range_query)
    buckets = {
        "xbox": OrderedDict((v, 0) for v in response['aggregations']['2']['buckets'].keys()),
        "ps": OrderedDict((v, 0) for v in response['aggregations']['2']['buckets'].keys()),
        "all": OrderedDict((v, 0) for v in response['aggregations']['2']['buckets'].keys()),
    }
    for key, value in response['aggregations']['2']['buckets'].items():
        buckets['all'][key] = value['doc_count']
        for bucket in value['3']['buckets']:
            buckets[bucket['key']][key] = bucket['doc_count']
    return buckets


def create_accounts_by_battles_chart(buckets, watermark_text='@WOTC_Tracker'):
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    then = datetime.utcnow() - timedelta(days=1)
    fig.suptitle("Breakdown of accounts by number of battles played for {}".format(then.strftime('%Y-%m-%d')))
    # ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=10)
    ax1 = plt.axes()
    ax1.axis('equal')
    size = 0.125

    outer_labels = []
    prev = 0
    for key in buckets['all'].keys():
        parts = key.split('-')
        outer_labels.append('{}-{} battles'.format(int(float(parts[0])) if parts[0] != '*' else parts[0], int(float(parts[1])) - 1 if parts[1] != '*' else parts[1]))

    # Outer pie chart
    outer_cmap = plt.get_cmap("binary")
    outer_colors = outer_cmap([i * 10 for i in range(10, len(buckets['all'].keys()) + 11)])
    outer_wedges, outer_text, outer_autotext = ax1.pie(
        buckets['all'].values(),
        explode=[0.1 for __ in outer_labels],
        radius=1,
        colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor='w'),
        autopct='%1.1f%%',
        pctdistance=1.1
        #labels=outer_labels
    )

    bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle='-'), bbox=bbox_props, zorder=0, va='center')
    for i, wedge in enumerate(outer_wedges):
        angle = calc_angle(wedge)
        y = sin(angle * (pi / 180))
        x = cos(angle * (pi / 180))
        align = 'right' if x < 0 else 'left'
        connectionstyle = 'angle,angleA=0,angleB={}'.format(angle)
        kw['arrowprops'].update({'connectionstyle': connectionstyle})
        ax1.annotate(
            outer_labels[i],
            xy=(x, y),
            xytext=(1.35*(-1 if x < 0 else 1), 1.4*y),
            horizontalalignment=align,
            **kw
        )

    # Inner pie chart
    inner_cmap = plt.get_cmap("tab20c")
    pie_flat = list(zip(buckets['xbox'].values(), buckets['ps'].values()))
    inner_labels = []
    for pair in pie_flat:
        inner_labels.extend(['xbox', 'ps'])
    inner_colors = inner_cmap([1 if console == 'ps' else 9 for console in inner_labels])
    inner_wedges, inner_text, inner_autotext = ax1.pie(
        [item for sublist in pie_flat for item in sublist],
        explode=[0.1 for __ in inner_labels],
        radius=1.05-size,
        colors=inner_colors,
        wedgeprops=dict(width=size, edgecolor='w'),
        autopct='',
        pctdistance=0.9
    )

    # Replace inner text with actual values
    for i, label, wedge, text in zip(range(len(inner_wedges)), inner_labels, inner_wedges, inner_autotext):
        text.set_text(buckets[label]['other' if i // 2 > len(buckets['all'].keys()) - 1 else list(buckets['all'].keys())[i // 2]])
        angle = calc_angle(wedge)
        if 90 < angle < 270:
            angle += 180
        text.set_rotation(angle)

    # Patch inner wedges to group together in explosion
    # Influenced by: https://stackoverflow.com/a/20556088/1993468
    groups = [[i, i+1] for i in range(0, len(inner_wedges), 2)]
    radfraction = 0.1
    for group in groups:
        angle = ((inner_wedges[group[-1]].theta2 + inner_wedges[group[0]].theta1)/2) * (pi / 180)
        for g in group:
            wedge = inner_wedges[g]
            wedge.set_center((radfraction * wedge.r * cos(angle), radfraction * wedge.r * sin(angle)))

    ax1.legend(inner_wedges[-2:], ['xbox', 'ps'], loc='lower right')
    fig.text(0.5, 0.5, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(BATTLERANGE_PNG)
    del fig


def query_five_battles_a_day_minimum(config):
    now = datetime.utcnow()
    then = now - timedelta(days=config['days'])
    es = Elasticsearch(**config['elasticsearch'])
    five_battles_a_day_query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    five_battles_a_day_query['query']['bool']['must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    response = es.search(index=config['battle index'], body=five_battles_a_day_query)

    buckets = {
        "xbox": OrderedDict(),
        "ps": OrderedDict(),
        "all": OrderedDict()
    }

    for bucket in response['aggregations']['4']['buckets']:
        key = bucket['key_as_string'].split('T')[0]
        buckets['xbox'][key] = 0
        buckets['ps'][key] = 0
        buckets['all'][key] = 0
        for subbucket in bucket['3']['buckets']:
            buckets[subbucket['key']][key] = subbucket['2']['buckets']['5.0-*']['doc_count']
        buckets['all'][key] = buckets['xbox'][key] + buckets['ps'][key]

    return buckets


# Requested by Khorne Dog in the forums
def create_five_battles_minimum_chart(buckets, watermark_text='@WOTC_Tracker'):
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150)
    fig.suptitle("Number of accounts having played at least 5 battles")
    ax1 = fig.add_subplot(111)

    width = 0.25
    keys = [datetime.strptime(d, '%Y-%m-%d') - timedelta(days=1) for d in buckets['all'].keys()]
    xkeys = [d - timedelta(hours=3) for d in keys]
    pkeys = [d + timedelta(hours=3) for d in keys]
    xbox_bars = ax1.bar(xkeys, buckets['xbox'].values(), width=width, color='g')
    ps_bars = ax1.bar(pkeys, buckets['ps'].values(), width=width, color='b')
    ax1.table(
        cellText=[
            list(buckets['xbox'].values()),
            list(buckets['ps'].values()),
            list(buckets['all'].values())],
        rowLabels=['xbox', 'ps', 'all'],
        colLabels=[d.strftime('%Y-%m-%d') for d in keys],
        loc='bottom')
    ax1.set_ylabel('Accounts')
    ax1.set_xticks([])
    ax1.legend((xbox_bars[0], ps_bars[0]), ('xbox', 'ps'))
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(FIVEADAY_PNG)


def query_long_term_data(config, filter_server_failures=True):
    now = datetime.utcnow()
    then = now - timedelta(days=config.get('long term', 90) + 1)
    es = Elasticsearch(**config['elasticsearch'])
    # Setup queries
    battles_query['query']['bool'][
        'must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    battles_query['query']['bool'][
        'must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    players_query['query']['bool'][
        'must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    players_query['query']['bool'][
        'must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')

    players = es.search(index=config['battle index'], body=players_query)
    battles = es.search(index=config['battle index'], body=battles_query)

    players_buckets = {
        "xbox": OrderedDict(),
        "ps": OrderedDict(),
        "all": OrderedDict()
    }

    battles_buckets = {
        "xbox": OrderedDict(),
        "ps": OrderedDict(),
        "all": OrderedDict()
    }

    average_battles_per_day_buckets = {
        "xbox": OrderedDict(),
        "ps": OrderedDict(),
        "all": OrderedDict()
    }

    for bucket in players['aggregations']['2']['buckets']:
        key = bucket['key_as_string'].split('T')[0]
        players_buckets['xbox'][key] = 0
        players_buckets['ps'][key] = 0
        players_buckets['all'][key] = 0
        if not bucket['3']['buckets']:
            continue
        for subbucket in bucket['3']['buckets']:
            players_buckets[subbucket['key']][key] = subbucket['doc_count']
        players_buckets['all'][key] = players_buckets['xbox'][key] + players_buckets['ps'][key]

    for bucket in battles['aggregations']['2']['buckets']:
        key = bucket['key_as_string'].split('T')[0]
        battles_buckets['xbox'][key] = 0
        battles_buckets['ps'][key] = 0
        battles_buckets['all'][key] = 0
        if not bucket['3']['buckets']:
            continue
        for subbucket in bucket['3']['buckets']:
            battles_buckets[subbucket['key']][key] = subbucket['1']['value']
        battles_buckets['all'][key] = battles_buckets['xbox'][key] + battles_buckets['ps'][key]

    if filter_server_failures:
        skip_next = False
        for key, value in players_buckets['ps'].items():
            # 20,000 is way below normal. Sometimes the server dies partway through. This day should be skipped
            if value < 20000:
                players_buckets['xbox'][key] = None
                players_buckets['ps'][key] = None
                players_buckets['all'][key] = None
                battles_buckets['xbox'][key] = None
                battles_buckets['ps'][key] = None
                battles_buckets['all'][key] = None
                skip_next = True
            elif skip_next:
                players_buckets['xbox'][key] = None
                players_buckets['ps'][key] = None
                players_buckets['all'][key] = None
                battles_buckets['xbox'][key] = None
                battles_buckets['ps'][key] = None
                battles_buckets['all'][key] = None
                skip_next = False

    for key in players_buckets['all'].keys():
        if players_buckets['xbox'][key] is None:
            average_battles_per_day_buckets['all'][key] = None
            average_battles_per_day_buckets['xbox'][key] = None
            average_battles_per_day_buckets['ps'][key] = None
        else:
            average_battles_per_day_buckets['xbox'][key] = battles_buckets['xbox'][key] / players_buckets['xbox'][key]
            average_battles_per_day_buckets['ps'][key] = battles_buckets['ps'][key] / players_buckets['ps'][key]
            average_battles_per_day_buckets['all'][key] = (battles_buckets['xbox'][key] + battles_buckets['ps'][key]) / (players_buckets['xbox'][key] + players_buckets['ps'][key])

    delkey = list(players_buckets['all'].keys())[0]
    # delkey = list(battles_buckets['all'].keys())[0]
    del players_buckets['all'][key]
    del players_buckets['xbox'][key]
    del players_buckets['ps'][key]
    del battles_buckets['all'][key]
    del battles_buckets['xbox'][key]
    del battles_buckets['ps'][key]
    del average_battles_per_day_buckets['xbox'][key]
    del average_battles_per_day_buckets['ps'][key]
    del average_battles_per_day_buckets['all'][key]

    return players_buckets, battles_buckets, average_battles_per_day_buckets


def create_long_term_charts(players_buckets, battles_buckets, average_battles_per_day_buckets, watermark_text='@WOTC_Tracker'):
    dates = [datetime.strptime(d, '%Y-%m-%d') - timedelta(days=1) for d in players_buckets['all'].keys()]
    # Players PNG
    plt.clf()
    fig = plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Active Accounts Per Platform (long view)')
    ax1 = fig.add_subplot(111)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.plot(dates, players_buckets['xbox'].values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, players_buckets['ps'].values(), color='blue', linewidth=2, label='Playstation')
    ax1.set_xticks(dates)
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, -0.15, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.tight_layout()
    fig.autofmt_xdate()
    # ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    fig.savefig(PLAYERSLONG_PNG)
    del fig
    # Battles PNG
    plt.clf()
    fig = plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Total Battles Per Platform (long view)')
    ax1 = fig.add_subplot(111)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.plot(dates, battles_buckets['xbox'].values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, battles_buckets['ps'].values(), color='blue', linewidth=2, label='Playstation')
    ax1.set_xticks(dates)
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, -0.15, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.tight_layout()
    fig.autofmt_xdate()
    # ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    fig.savefig(BATTLESLONG_PNG)
    del fig
    # Average PNG
    plt.clf()
    fig = plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Average Battles Played Per Account Per Platform (long view)')
    ax1 = fig.add_subplot(111)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.plot(dates, average_battles_per_day_buckets['xbox'].values(), color='green', linewidth=2, label='Xbox')
    ax1.plot(dates, average_battles_per_day_buckets['ps'].values(), color='blue', linewidth=2, label='Playstation')
    ax1.set_xticks(dates)
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, -0.15, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.tight_layout()
    fig.autofmt_xdate()
    # ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    fig.savefig(AVERAGELONG_PNG)
    del fig


def upload_long_term_charts(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    playerslong = api.media_upload(PLAYERSLONG_PNG)
    battleslong = api.media_upload(BATTLESLONG_PNG)
    averagelong = api.media_upload(AVERAGELONG_PNG)
    api.update_status(
        status='Long-term view of active accounts, with downtime and multi-day catchup errors omitted',
        media_ids=[playerslong.media_id, battleslong.media_id, averagelong.media_id]
    )


def upload_long_term_mode_charts(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    modelong = api.media_upload(MODEBREAKDOWNLONG_PNG)
    percentlong = api.media_upload(MODEBREAKDOWNPERCENTLONG_PNG)
    api.update_status(
        status='Long-term view of battles per mode',
        media_ids=[modelong.media_id, percentlong.media_id]
    )


def upload_activity_graphs_to_twitter(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    battles = api.media_upload(BATTLES_PNG)
    players = api.media_upload(PLAYERS_PNG)
    newplayers = api.media_upload(NEWPLAYERS_PNG)
    averages = api.media_upload(AVERAGE_PNG)
    api.update_status(
        status=config['twitter']['message'],
        media_ids=[players.media_id, battles.media_id, newplayers.media_id, averages.media_id]
    )


def upload_account_age_graph_to_twitter(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    accountage = api.media_upload(ACCOUNTAGE_PNG)
    api.update_status(
        status='Breakdown of active accounts by age per platform on #worldoftanksconsole',
        media_ids=[accountage.media_id]
    )


def upload_accounts_by_battles_chart_to_twitter(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    battlerange = api.media_upload(BATTLERANGE_PNG)
    api.update_status(
        status='Breakdown of accounts by number of battles played on #worldoftanksconsole',
        media_ids=[battlerange.media_id]
    )


def upload_five_battles_minimum_chart_to_twitter(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    fiveaday = api.media_upload(FIVEADAY_PNG)
    api.update_status(
        status='Filtering accounts per day with 5 battles minimum on #worldoftanksconsole',
        media_ids=[fiveaday.media_id]
    )


def share_unique_with_twitter(config, unique):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    status = 'Unique Active Accounts For {} Over Time\n{}'
    formatting = '{} days: {}'
    for key, values in unique.items():
        api.update_status(
            status=status.format(
                key,
                '\n'.join(map(lambda l: formatting.format(
                    config['unique'][values.index(l)], l), values))
            )
        )


def build_cw_tanks_list(config):
    api = 'https://api-console.worldoftanks.com/wotx/encyclopedia/vehicles/'
    params = {
        'application_id': config['wg api key'],
        'fields': 'era,tank_id'
    }
    data = get(api, params=params).json()['data']
    return ' OR '.join(
        list(
            map(
                lambda t: 'tank_id:{}'.format(t['tank_id']),
                filter(lambda t: t['era'] != '', data.values())
                )
            )
        )


def query_es_for_top_tanks(config, era):
    now = datetime.utcnow()
    then = now - timedelta(days=1)
    es = Elasticsearch(**config['elasticsearch'])
    if era == 'ww2':
        query = ww2_popular_tanks_query
    elif era == 'cw':
        query = cw_popular_tanks_query
    # Setup query
    query['query']['bool']['must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    # Query Elasticsearch
    response = es.search(index=config['tank index'], body=query)
    buckets = {
        'xbox': OrderedDict(),
        'ps': OrderedDict()
    }
    for bucket in response['aggregations']['2']['buckets']:
        for subbucket in bucket['4']['buckets']:
            key = subbucket['key']
            for tank in subbucket['3']['buckets']:
                buckets[key][tank['key']] = int(tank['1']['value'])
    return buckets


def query_for_tank_info(tanks):
    url = 'https://wotconsole.ru/api/tankopedia/en/{}.json'
    new_tanks = {
        'xbox': OrderedDict(),
        'ps': OrderedDict()
    }
    for plat, t in tanks.items():
        for tank, battles in t.items():
            response = get(url.format(tank))
            new_tanks[plat][response.json()['info']['user_string']] = battles
    new_tanks['playstation'] = new_tanks['ps']
    del new_tanks['ps']
    return new_tanks


def share_top_tanks(config, era, top, day):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    for platform, tanks in top.items():
        status = "Most used {} tanks on {} for {}\n{}"
        formatting = '{}: {} battles'
        api.update_status(
            status=status.format(
                era,
                platform.capitalize(),
                day,
                '\n'.join([formatting.format(tank, battles) for tank, battles in tanks.items()])
            )
        )


def query_es_for_mode_battles_difference(config, long_term=False):
    now = datetime.utcnow()
    then = now - timedelta(days=config['days'] if not long_term else config['long term'])
    es = Elasticsearch(**config['elasticsearch'])
    # Setup query
    battles_query['query']['bool']['must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    battles_query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    cw_popular_tanks_query['query']['bool']['must'][-1]['range']['date']['gte'] = then.strftime('%Y-%m-%d')
    cw_popular_tanks_query['query']['bool']['must'][-1]['range']['date']['lte'] = now.strftime('%Y-%m-%d')
    # Query Elasticsearch
    total_battles_response = es.search(index=config['battle index'], body=battles_query)
    cw_battles_response = es.search(index=config['tank index'], body=cw_popular_tanks_query)
    dates = [b['key_as_string'].split('T')[0] for b in total_battles_response[
        'aggregations']['2']['buckets']]
    # Filter numbers
    ww2_battles_xbox = OrderedDict()
    ww2_battles_ps = OrderedDict()
    cw_battles_xbox = OrderedDict()
    cw_battles_ps = OrderedDict()
    percent_cw_xbox = OrderedDict()
    percent_cw_ps = OrderedDict()
    for d in dates:
        ww2_battles_xbox[d] = 0
        ww2_battles_ps[d] = 0
        cw_battles_xbox[d] = 0
        cw_battles_ps[d] = 0
        percent_cw_xbox[d] = None
        percent_cw_ps[d] = None
    for bucket in total_battles_response['aggregations']['2']['buckets']:
        if not bucket['3']['buckets']:
            continue
        for subbucket in bucket['3']['buckets']:
            if subbucket['key'] == 'xbox':
                ww2_battles_xbox[bucket['key_as_string'].split('T')[0]] = subbucket['1']['value']
            else:
                ww2_battles_ps[bucket['key_as_string'].split('T')[0]] = subbucket['1']['value']
    for bucket in cw_battles_response['aggregations']['2']['buckets']:
        if not bucket['4']['buckets']:
            continue
        for subbucket in bucket['4']['buckets']:
            if subbucket['key'] == 'xbox':
                cw_battles_xbox[bucket['key_as_string'].split('T')[0]] = subbucket['1']['value']
            else:
                cw_battles_ps[bucket['key_as_string'].split('T')[0]] = subbucket['1']['value']
    for i in range(len(dates)):
        percent_cw_xbox[dates[i]] = cw_battles_xbox[dates[i]] / ww2_battles_xbox[dates[i]]
        percent_cw_ps[dates[i]] = cw_battles_ps[dates[i]] / ww2_battles_ps[dates[i]]
        ww2_battles_xbox[dates[i]] = ww2_battles_xbox[dates[i]] - cw_battles_xbox[dates[i]]
        ww2_battles_ps[dates[i]] = ww2_battles_ps[dates[i]] - cw_battles_ps[dates[i]]
    return dates, list(ww2_battles_xbox.values()), list(ww2_battles_ps.values()), list(cw_battles_xbox.values()), list(cw_battles_ps.values()), list(percent_cw_xbox.values()), list(percent_cw_ps.values())


def create_mode_difference_graph(dates, ww2_battles_xbox, ww2_battles_ps, cw_battles_xbox, cw_battles_ps, percent_cw_xbox, percent_cw_ps, long_term=False, watermark_text='@WOTC_Tracker'):
    shifted_dates = [(datetime.strptime(d, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d') for d in dates]
    # Mode PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150) if not long_term else plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Estimated breakdown of battles between CW and WW2, per platform' if not long_term else 'Estimated breakdown of battles between CW and WW2, per platform (long term)')
    # ax1 = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.set_xticklabels(shifted_dates, ha='right')
    ax1.plot(shifted_dates, ww2_battles_xbox, color='darkgreen', linewidth=2, label='WW2: Xbox')
    ax1.plot(shifted_dates, cw_battles_xbox, color='lightgreen', linewidth=2, label='CW: Xbox')
    ax1.plot(shifted_dates, ww2_battles_ps, color='darkblue', linewidth=2, label='WW2: Playstation')
    ax1.plot(shifted_dates, cw_battles_ps, color='lightblue', linewidth=2, label='CW: Playstation')
    ax1.set_ylim(bottom=0)
    # for i in range(len(shifted_dates)):
    #     xbox_text = ax1.annotate(annotations_xbox[i], (shifted_dates[i], ww2_battles_xbox[i]), verticalalignment='bottom', size=12 if not long_term else 8)
    #     ps_text = ax1.annotate(annotations_ps[i], (shifted_dates[i], ww2_battles_ps[i]), verticalalignment='bottom', size=12 if not long_term else 8)
    #     xbox_text.set_rotation(90)
    #     ps_text.set_rotation(90)
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(MODEBREAKDOWN_PNG if not long_term else MODEBREAKDOWNLONG_PNG)
    del fig
    # Mode Percent PNG
    plt.clf()
    fig = plt.figure(figsize=(11, 8), dpi=150) if not long_term else plt.figure(figsize=(24, 8), dpi=150)
    fig.suptitle('Estimated percentage of battles taking place in CW, per platform' if not long_term else 'Estimated percentage of battles taking place in CW, per platform (long term)')
    # ax1 = plt.axes()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.set_xticklabels(shifted_dates, ha='right')
    ax1.plot(shifted_dates, percent_cw_xbox, color='green', linewidth=2, label='Xbox')
    ax1.plot(shifted_dates, percent_cw_ps, color='blue', linewidth=2, label='Playstation')
    ax1.grid()
    ax1.legend()
    ax1.text(0.5, 1.05, watermark_text, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(MODEBREAKDOWNPERCENT_PNG if not long_term else MODEBREAKDOWNPERCENTLONG_PNG)
    del fig


def upload_mode_breakdown_to_twitter(config):
    auth = OAuthHandler(
        config['twitter']['api key'],
        config['twitter']['api secret key'])
    auth.set_access_token(
        config['twitter']['access token'],
        config['twitter']['access token secret'])
    api = API(auth)
    battles = api.media_upload(MODEBREAKDOWN_PNG)
    percent = api.media_upload(MODEBREAKDOWNPERCENT_PNG)
    api.update_status(
        status="Estimated split between WW2 and CW battles",
        media_ids=[battles.media_id, percent.media_id]
    )


def get_universal_params(config):
    params = dict()
    watermark = config.get('watermark text', None)
    if watermark:
        params['watermark_text'] = watermark
    return params


if __name__ == '__main__':
    agp = ArgumentParser(
        description='Bot for processing tracker data and uploading to Twitter')
    agp.add_argument('config', help='Config file location')
    agp.add_argument('-u', '--upload', help='Upload to twitter', action='store_true')
    agp.add_argument('--activity-graphs', action='store_true')
    agp.add_argument('--account-age', action='store_true')
    agp.add_argument('--accounts-by-battles', action='store_true')
    agp.add_argument('--five-battles-min', action='store_true')
    agp.add_argument('--long-term', action='store_true')
    agp.add_argument('--share-unique', action='store_true')
    agp.add_argument('--top-cw-tanks', action='store_true')
    agp.add_argument('--top-ww2-tanks', action='store_true')
    agp.add_argument('--mode-breakdown', action='store_true')
    args = agp.parse_args()
    config = manage_config('read', args.config)
    additional_params = get_universal_params(config)
    now = datetime.utcnow()
    if args.top_cw_tanks or args.top_ww2_tanks or args.mode_breakdown or args.long_term:
        CW_TANKS = build_cw_tanks_list(config)
        cw_popular_tanks_query['query']['bool']['must'][0]['query_string']['query'] = CW_TANKS
        ww2_popular_tanks_query['query']['bool']['must'][0]['query_string']['query'] = 'NOT (' + CW_TANKS + ')'
    if args.activity_graphs:
        try:
            create_activity_graphs(*query_es_for_graphs(config), **additional_params)
            if args.upload:
                upload_activity_graphs_to_twitter(config)
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.account_age:
        try:
            create_account_age_chart(query_es_for_active_accounts(config), **additional_params)
            if args.upload:
                upload_account_age_graph_to_twitter(config)
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.accounts_by_battles:
        try:
            create_accounts_by_battles_chart(query_es_for_accounts_by_battles(config), **additional_params)
            if args.upload:
                upload_accounts_by_battles_chart_to_twitter(config)
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.five_battles_min:
        try:
            create_five_battles_minimum_chart(query_five_battles_a_day_minimum(config), **additional_params)
            if args.upload:
                upload_five_battles_minimum_chart_to_twitter(config)
        except Exception as e:
            # print(e)
            traceback.print_exc()
    # Limit long-term views to beginning of month to review previous month's history
    if args.long_term:
        if now.day == 1:
            try:
                create_long_term_charts(*query_long_term_data(config, config.get('omit errors long term', True)), **additional_params)
                create_mode_difference_graph(*query_es_for_mode_battles_difference(config, long_term=True), long_term=True, **additional_params)
                if args.upload:
                    upload_long_term_charts(config)
                    upload_long_term_mode_charts(config)
            except Exception as e:
                # print(e)
                traceback.print_exc()
    if args.share_unique:
        try:
            share_unique_with_twitter(config, query_es_for_unique(config))
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.top_cw_tanks:
        try:
            share_top_tanks(config, 'CW', query_for_tank_info(query_es_for_top_tanks(config, 'cw')), (now - timedelta(days=1)).strftime('%Y-%m-%d'))
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.top_ww2_tanks:
        try:
            share_top_tanks(config, 'WW2', query_for_tank_info(query_es_for_top_tanks(config, 'ww2')), (now - timedelta(days=1)).strftime('%Y-%m-%d'))
        except Exception as e:
            # print(e)
            traceback.print_exc()
    if args.mode_breakdown:
        try:
            create_mode_difference_graph(*query_es_for_mode_battles_difference(config), **additional_params)
            if args.upload:
                upload_mode_breakdown_to_twitter(config)
        except Exception as e:
            # print(e)
            traceback.print_exc()
