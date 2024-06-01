from marshmallow import fields

from .base import BaseSchema


class ScheduledAnalysisSchema(BaseSchema):
    analysis_system_instance = fields.Url(required=True)
    sample = fields.Url(required=True)
    analysis_scheduled = fields.DateTime(required=True)
    priority = fields.Int(required=True)
