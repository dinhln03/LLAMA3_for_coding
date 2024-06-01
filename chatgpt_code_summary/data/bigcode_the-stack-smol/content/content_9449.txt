from sendbee_api.models import Model
from sendbee_api.fields import TextField, BooleanField


class RateLimitError(Model):
    """Data model for rate limit error"""

    _detail = TextField(index='detail', desc='Message detail')
    _error = BooleanField(index='error', desc='Error or not')
    _type = TextField(index='type', desc='Message type')
