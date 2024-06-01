from rest_framework import serializers

from apps.currency.models import Currency


class CurrencyWalletSerializer(serializers.ModelSerializer):
    actual_nonce = serializers.SerializerMethodField("get_nonce")

    def get_nonce(self, wallet):
        return wallet.nonce

    class Meta:
        from apps.wallet.models import Wallet

        model = Wallet
        fields = ["wallet_id", "public_key", "actual_nonce", "category", "state"]


class CurrencySerializer(serializers.ModelSerializer):
    owner_wallet = CurrencyWalletSerializer(source="cashout_wallet")
    owner_wallet_new = CurrencyWalletSerializer(source="owner_wallet")
    cashout_wallet = CurrencyWalletSerializer()

    class Meta:
        model = Currency
        fields = [
            "uuid",
            "name",
            "symbol",
            "token_id",
            "decimals",
            "campaign_end",
            "claim_deadline",
            "allow_minting",
            "owner_wallet_new",
            "owner_wallet",
            "cashout_wallet",
            "starting_capital",
            "is_public",
            "needs_sms_verification",
        ]
