from rest_framework import serializers
from rest_framework.reverse import reverse

from .models import Channel, Datasource, Value


class ChannelSerializer(serializers.HyperlinkedModelSerializer):

    values = serializers.SerializerMethodField()
    latest = serializers.SerializerMethodField()

    def get_values(self, obj):
        request = self.context.get("request")
        return request.build_absolute_uri(
            reverse(
                "datasource-channel-values-list",
                kwargs={"datasource_pk": obj.datasource_id, "channel_pk": obj.id},
            )
        )

    def get_latest(self, obj):
        request = self.context.get("request")
        return request.build_absolute_uri(
            reverse(
                "datasource-channel-values-latest",
                kwargs={"datasource_pk": obj.datasource_id, "channel_pk": obj.id},
            )
        )

    class Meta:
        model = Channel
        fields = ("id", "uniquename", "name", "values", "latest")


class ValueSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Value
        fields = ("id", "time", "value")


class DatasourceSerializer(serializers.HyperlinkedModelSerializer):
    channels = serializers.HyperlinkedIdentityField(
        view_name="datasource-channels-list", lookup_url_kwarg="datasource_pk"
    )

    class Meta:
        model = Datasource
        fields = ("id", "devid", "name", "description", "lat", "lon", "channels")
