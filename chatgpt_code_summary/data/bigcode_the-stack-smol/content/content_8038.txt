from currency_exchanger.currencies.models import Currency
from currency_exchanger.wallets.models import Wallet
from django.db import models


class Stock(models.Model):
    symbol = models.CharField(max_length=10)
    currency = models.ForeignKey(Currency, on_delete=models.CASCADE, related_name="stocks")
    price = models.DecimalField(decimal_places=2, max_digits=10)

    def __str__(self):
        return self.symbol


class WalletStock(models.Model):
    wallet = models.ForeignKey(Wallet, on_delete=models.CASCADE)
    stocks = models.ForeignKey(Stock, on_delete=models.CASCADE)
    count = models.PositiveIntegerField(default=0)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["wallet", "stocks"], name="unique_wallet_stock")
        ]


class StockTransfer(models.Model):
    wallet = models.ForeignKey(Wallet, on_delete=models.CASCADE, related_name="stock_transfers")
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name="+")
    amount = models.IntegerField()


class StockHistory(models.Model):
    stocks = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name="history")
    timestamp = models.DateTimeField(auto_now_add=True)
    price = models.DecimalField(decimal_places=2, max_digits=10)

    class Meta:
        ordering = ["-timestamp"]
