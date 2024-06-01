###############################################################################
# 1)    This is a test case to verify that the deposit case works fine.
# 2)    It also checks whether duplicate requests are processed correctly.
###############################################################################

####################################
# Client Settings
# The client configuration is a dictionary where each key is a list of 
# all the clients of a particular bank. Each entry in the list is a key:value 
# pair of all the configurations of that client
####################################
client_conf = { 'CITI': 
[
  {'index':0, 'account_no': 9999,'client_time_out': 8, 'num_retransmits':3, 'resend_to_new_head':1, 'msg_loss_freq':0},
],}

#The clients will issue the following requests in that order to the servers
client_seq = [('getBalance', ('UID1', 8888)),
              ('deposit', ('UID1', 8888, 100)),
              ('deposit', ('UID2', 8888, 100)),
              ('deposit', ('UID3', 8888, 100)),
              ('deposit', ('UID4', 8888, 100)),
              ('deposit', ('UID5', 8888, 100)),
              ('withdraw', ('UID6', 8888, 100)),
              ('withdraw', ('UID7', 8888, 100)),
              ('withdraw', ('UID8', 8888, 100)),
              ('withdraw', ('UID9', 8888, 100)),
              ('withdraw', ('UID10', 8888, 100)),
              ('getBalance', ('UID1', 8888))
              ]

#random(seed, numReq, probGetBalance, probDeposit, probWithdraw, probTransfer)
#client_prob_conf = [
#{'index':0, 'seed':450, 'numReq':10, 'prob':[('getBalance',0.10), ('deposit',0.5), ('withdraw',0.4), ('transfer',0)]}
#]

####################################
# Server Settings
# The server configuration is a dictionary where each key is a list of 
# all the servers of a particular bank. Each entry in the list is a key:value 
# pair of all the configurations of that server
####################################
server_conf = { 'CITI': 
[
    {'index':0, 'startup_delay': 0, 'rcv_lifetime':1000, 'snd_lifetime':1000, 'ip_addr': '127.0.0.1', 'port': 1001, 'heartbeat_interval':1},
    {'index':1, 'startup_delay': 13, 'rcv_lifetime':1000, 'snd_lifetime':1000, 'ip_addr': '127.0.0.1', 'port': 1002, 'heartbeat_interval':1},
    {'index':2, 'startup_delay': 0, 'rcv_lifetime':1000, 'snd_lifetime':1000, 'ip_addr': '127.0.0.1', 'port': 1003, 'heartbeat_interval':1}
],}

master_conf = { 'master_interval':5}
