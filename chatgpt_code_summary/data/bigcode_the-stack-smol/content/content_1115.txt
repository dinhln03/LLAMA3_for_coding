from prometheus_client import start_http_server, Gauge, Counter

all_users = Gauge('users_in_all_guilds', 'All users the bot is able to see.')
all_guilds = Gauge('guilds_bot_is_in', 'The amount of guilds the bot is in.')

ready_events = Counter('ready_events', 'Amount of READY events recieved during uptime.')
message_events = Counter('message_events', 'Amount of messages sent during uptime.')
reconnects = Counter('reconnects', 'Amount of reconnects the bot has done to Discords API.')

def startup_prometheus():
    start_http_server(9091)
