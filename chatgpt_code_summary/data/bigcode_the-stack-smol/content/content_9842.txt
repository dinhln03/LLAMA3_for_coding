import logging
from korbit.client.korbit_client import KorbitClient

logging.basicConfig(level=logging.INFO)

properties_sandbox_file = '../properties_sandbox_test.json'
context_sandbox_file = '../context_sandbox.json'

kbclient = KorbitClient(properties_sandbox_file, context_sandbox_file)

print(kbclient.getUserInfo())

# 매수 Buy
# print( kbclient.buy(price=300000, coin_amount=1) )
# # 매도 Sell
# print( kbclient.sell(price=300000, coin_amount=1) )
print( kbclient.getOpenOrders() )


# Wallet Test
wallet = kbclient.getWallet()
balance = wallet['balance']
pending_orders = wallet['pendingOrders']
available = wallet['available']

print(balance)
print(pending_orders)
print(available)