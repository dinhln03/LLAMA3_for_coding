from itertools import islice
from tests.unit.utils import Teardown

import inspect
import pytest
import time

import cbpro.messenger
import cbpro.public
import cbpro.private


class TestPrivateClient(object):
    def test_private_attr(self, private_client):
        assert isinstance(private_client, cbpro.public.PublicClient)
        assert hasattr(private_client, 'accounts')
        assert hasattr(private_client, 'orders')
        assert hasattr(private_client, 'fills')
        assert hasattr(private_client, 'limits')
        assert hasattr(private_client, 'deposits')
        assert hasattr(private_client, 'withdrawals')
        assert hasattr(private_client, 'conversions')
        assert hasattr(private_client, 'payments')
        assert hasattr(private_client, 'coinbase')
        assert hasattr(private_client, 'fees')
        assert hasattr(private_client, 'reports')
        assert hasattr(private_client, 'profiles')
        assert hasattr(private_client, 'oracle')

    def test_private_accounts(self, private_client):
        accounts = private_client.accounts
        assert isinstance(accounts, cbpro.messenger.Subscriber)
        assert isinstance(accounts, cbpro.private.Accounts)
        assert hasattr(accounts, 'list')
        assert hasattr(accounts, 'get')
        assert hasattr(accounts, 'history')
        assert hasattr(accounts, 'holds')

    def test_private_orders(self, private_client):
        orders = private_client.orders
        assert isinstance(orders, cbpro.messenger.Subscriber)
        assert isinstance(orders, cbpro.private.Orders)
        assert hasattr(orders, 'post')
        assert hasattr(orders, 'cancel')
        assert hasattr(orders, 'list')
        assert hasattr(orders, 'get')

    def test_private_fills(self, private_client):
        fills = private_client.fills
        assert isinstance(fills, cbpro.messenger.Subscriber)
        assert isinstance(fills, cbpro.private.Fills)
        assert hasattr(fills, 'list')

    def test_private_limits(self, private_client):
        limits = private_client.limits
        assert isinstance(limits, cbpro.messenger.Subscriber)
        assert isinstance(limits, cbpro.private.Limits)
        assert hasattr(limits, 'get')

    def test_private_deposits(self, private_client):
        deposits = private_client.deposits
        assert isinstance(deposits, cbpro.messenger.Subscriber)
        assert isinstance(deposits, cbpro.private.Deposits)
        assert hasattr(deposits, 'list')
        assert hasattr(deposits, 'get')
        assert hasattr(deposits, 'payment')
        assert hasattr(deposits, 'coinbase')
        assert hasattr(deposits, 'generate')

    def test_private_withdrawals(self, private_client):
        withdrawals = private_client.withdrawals
        assert isinstance(withdrawals, cbpro.messenger.Subscriber)
        assert isinstance(withdrawals, cbpro.private.Deposits)
        assert isinstance(withdrawals, cbpro.private.Withdrawals)
        assert hasattr(withdrawals, 'list')
        assert hasattr(withdrawals, 'get')
        assert hasattr(withdrawals, 'payment')
        assert hasattr(withdrawals, 'coinbase')
        assert hasattr(withdrawals, 'generate')
        assert hasattr(withdrawals, 'crypto')
        assert hasattr(withdrawals, 'estimate')

    def test_private_conversions(self, private_client):
        conversions = private_client.conversions
        assert isinstance(conversions, cbpro.messenger.Subscriber)
        assert isinstance(conversions, cbpro.private.Conversions)
        assert hasattr(conversions, 'post')

    def test_private_payments(self, private_client):
        payments = private_client.payments
        assert isinstance(payments, cbpro.messenger.Subscriber)
        assert isinstance(payments, cbpro.private.Payments)
        assert hasattr(payments, 'list')

    def test_private_coinbase(self, private_client):
        coinbase = private_client.coinbase
        assert isinstance(coinbase, cbpro.messenger.Subscriber)
        assert isinstance(coinbase, cbpro.private.Coinbase)
        assert hasattr(coinbase, 'list')

    def test_private_fees(self, private_client):
        fees = private_client.fees
        assert isinstance(fees, cbpro.messenger.Subscriber)
        assert isinstance(fees, cbpro.private.Fees)
        assert hasattr(fees, 'list')

    def test_private_reports(self, private_client):
        reports = private_client.reports
        assert isinstance(reports, cbpro.messenger.Subscriber)
        assert isinstance(reports, cbpro.private.Reports)

    def test_private_profiles(self, private_client):
        profiles = private_client.profiles
        assert isinstance(profiles, cbpro.messenger.Subscriber)
        assert isinstance(profiles, cbpro.private.Profiles)
        assert hasattr(profiles, 'list')
        assert hasattr(profiles, 'get')
        assert hasattr(profiles, 'transfer')

    def test_private_oracle(self, private_client):
        oracle = private_client.oracle
        assert isinstance(oracle, cbpro.messenger.Subscriber)
        assert isinstance(oracle, cbpro.private.Oracle)


@pytest.mark.skip
class TestPrivateAccounts(Teardown):
    def test_list(self, private_client):
        response = private_client.accounts.list()
        assert isinstance(response, list)
        assert 'currency' in response[0]

    def test_get(self, private_client, account_id):
        response = private_client.accounts.get(account_id)
        assert isinstance(response, dict)
        assert 'currency' in response

    def test_history(self, private_client, account_id):
        response = private_client.accounts.history(account_id)

        assert inspect.isgenerator(response)

        accounts = list(islice(response, 5))

        assert 'amount' in accounts[0]
        assert 'details' in accounts[0]

    def test_holds(self, private_client, account_id):
        response = private_client.accounts.holds(account_id)

        assert inspect.isgenerator(response)

        holds = list(islice(response, 5))

        assert 'type' in holds[0]
        assert 'ref' in holds[0]


@pytest.mark.skip
class TestPrivateOrders(Teardown):
    def test_post_limit_order(self, private_client, private_model):
        json = private_model.orders.limit('buy', 'BTC-USD', 40000.0, 0.001)
        response = private_client.orders.post(json)
        assert isinstance(response, dict)
        assert response['type'] == 'limit'

    def test_post_market_order(self, private_client, private_model):
        json = private_model.orders.market('buy', 'BTC-USD', size=0.001)
        response = private_client.orders.post(json)
        assert isinstance(response, dict)
        assert 'status' in response
        assert response['type'] == 'market'

    @pytest.mark.parametrize('stop', ['entry', 'loss'])
    def test_post_stop_order(self, private_client, private_model, stop):
        json = private_model.orders.market(
            'buy', 'BTC-USD', size=0.001, stop=stop, stop_price=30000
        )
        response = private_client.orders.post(json)
        assert isinstance(response, dict)
        assert response['stop'] == stop
        assert response['type'] == 'market'

    def test_cancel(self, private_client, private_model):
        json = private_model.orders.limit('buy', 'BTC-USD', 40000.0, 0.001)
        order = private_client.orders.post(json)
        time.sleep(0.2)
        params = private_model.orders.cancel('BTC-USD')
        response = private_client.orders.cancel(order['id'], params)
        assert isinstance(response, list)
        assert response[0] == order['id']

    def test_list(self, private_client, private_model):
        params = private_model.orders.list('pending')
        response = private_client.orders.list(params)

        assert inspect.isgenerator(response)

        orders = list(islice(response, 10))

        assert isinstance(orders, list)
        assert 'created_at' in orders[0]

    def test_get(self, private_client, private_model):
        json = private_model.orders.limit('buy', 'BTC-USD', 40000.0, 0.001)
        order = private_client.orders.post(json)
        time.sleep(0.2)
        response = private_client.orders.get(order['id'])
        assert response['id'] == order['id']


@pytest.mark.skip
class TestPrivateFills(Teardown):
    def test_list(self, private_client, private_model):
        params = private_model.fills.list('BTC-USD')
        response = private_client.fills.list(params)

        assert inspect.isgenerator(response)

        fills = list(islice(response, 10))

        assert isinstance(fills, list)
        assert 'fill_fees' in fills[0]


@pytest.mark.skip
class TestPrivateLimits(Teardown):
    def test_get(self, private_client):
        response = private_client.limits.get()
        assert isinstance(response, dict)


@pytest.mark.skip
class TestPrivateDeposits(Teardown):
    pass


@pytest.mark.skip
class TestPrivateWithdrawals(Teardown):
    pass


@pytest.mark.skip
class TestPrivateConversions(Teardown):
    def test_post(self, private_client, private_model):
        json = private_model.conversions.post('USD', 'USDC', 10.0)
        response = private_client.conversions.post(json)
        assert isinstance(response, dict)
        assert 'id' in response
        assert 'amount' in response
        assert response['from'] == 'USD'
        assert response['to'] == 'USDC'


@pytest.mark.skip
class TestPrivatePayments(Teardown):
    def test_list(self, private_client):
        response = private_client.payments.list()
        assert isinstance(response, list)


@pytest.mark.skip
class TestPrivateCoinbase(Teardown):
    def test_list(self, private_client):
        response = private_client.coinbase.list()
        assert isinstance(response, list)


@pytest.mark.skip
class TestPrivateFees(Teardown):
    def test_list(self, private_client):
        response = private_client.fees.list()
        assert isinstance(response, list)


@pytest.mark.skip
class TestPrivateReports(Teardown):
    pass


@pytest.mark.skip
class TestPrivateProfiles(Teardown):
    pass


@pytest.mark.skip
class TestPrivateOracle(Teardown):
    pass
