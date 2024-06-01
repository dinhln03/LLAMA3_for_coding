from django.core.exceptions import ObjectDoesNotExist
from rest_framework.serializers import PrimaryKeyRelatedField, RelatedField


class UniqueRelatedField(RelatedField):
    """
    Like rest_framework's PrimaryKeyRelatedField, but selecting by any unique
    field instead of the primary key.
    """

    default_error_messages = PrimaryKeyRelatedField.default_error_messages.copy()

    def __init__(self, field_name, serializer_field=None, **kwargs):
        super().__init__(**kwargs)
        self.related_field_name = field_name
        self.serializer_field = serializer_field

    def to_internal_value(self, data):
        if self.serializer_field is not None:
            data = self.serializer_field.to_internal_value(data)
        try:
            return self.get_queryset().get(**{self.related_field_name: data})
        except ObjectDoesNotExist:
            self.fail('does_not_exist', pk_value=data)
        except (TypeError, ValueError):
            self.fail('incorrect_type', data_type=type(data).__name__)

    def to_representation(self, value):
        value = getattr(value, self.related_field_name)
        if self.serializer_field is not None:
            value = self.serializer_field.to_representation(value)
        return value
