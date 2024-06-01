from rest_framework import serializers

from apps.challenge.models import Clan


class ClanSerializer(serializers.ModelSerializer):
    class Meta:
        model = Clan
        fields = (
            'name', 'leader', 'image', 'score', 'wins', 'losses', 'draws'
        )
