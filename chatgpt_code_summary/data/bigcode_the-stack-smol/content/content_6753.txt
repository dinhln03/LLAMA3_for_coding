from apis.creat_account.api_account_setAlias import account_setAlias
from apis.creat_account.api_create_account import create_account, create_account_100
from apis.creat_account.api_get_addresslist import get_address_list
from apis.transfer.blockmgr_sendRawTransaction import sendRawTransaction
from apis.transfer.time_of_account_1 import transation_120_account_1
from apis.transfer_inquiry.api_chain_getBalance import check_transfer_balance, transfer_balance, \
	getBalance_of_all_address_list, chain_getBalance
from apis.transfer.api_chain_transaction import transaction_one, random_transaction
from apis.transfer_inquiry.trace_getRawTransaction import getRawTransaction, getTransaction, decodeTrasnaction, \
	getReceiveTransactionByAd, rebuild, getSendTransactionByAddr
from apis.vote_message.account_voteCredit import voteCredit
from apis.vote_message.chain_getCreditDetails import getVoteCreditDetails
from apis.交易池中的交易状态及交易池中的交易流转过程.blockmgr_getPoolTransactions import getPoolTransactions
from apis.交易池中的交易状态及交易池中的交易流转过程.blockmgr_getTransactionCount import getTransactionCount
from apis.交易池中的交易状态及交易池中的交易流转过程.blockmgr_getTxInPool import blockmgrGetTxInPool

api_route = {
	"create_account": create_account,
	"create_account_100": create_account_100,
	"get_address_list": get_address_list,
	"account_setAlias": account_setAlias,
	"transaction_one": transaction_one,
	"random_transaction": random_transaction,
	"chain_getBalance": chain_getBalance,
	"getBalance_of_all_address_list": getBalance_of_all_address_list,
	# "creat_one_wallet_account": creat_one_wallet_account,
	"transation_120_account_1": transation_120_account_1,
	"transfer_balance": transfer_balance,
	"check_transfer_balance": check_transfer_balance,
	"getRawTransaction": getRawTransaction,
	"getTransaction": getTransaction,
	"decodeTrasnaction": decodeTrasnaction,
	"getSendTransactionByAddr": getSendTransactionByAddr,
	"getReceiveTransactionByAd": getReceiveTransactionByAd,
	"rebuild": rebuild,
	"blockmgrGetTxInPool": blockmgrGetTxInPool,
	"getPoolTransactions": getPoolTransactions,
	"getTransactionCount": getTransactionCount,
	"blockmgr_sendRawTransaction": sendRawTransaction,
	"account_voteCredit": voteCredit,
	"chain_getVoteCreditDetails": getVoteCreditDetails,
	
	
	
}


# API 总函数
def runCase(case_name):
	"""
	
	:param case_name:
	:return: 注意格式   xxx(case_name)()
	"""
	return api_route.get(case_name)()


if __name__ == '__main__':
	print(runCase("create_account_100"))
	print(runCase("create_account"))
