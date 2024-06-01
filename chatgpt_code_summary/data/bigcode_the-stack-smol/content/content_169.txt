from marshmallow import Schema, fields
from marshmallow.validate import OneOf

ticket_type = ("Bug", "Report", "Feature", "Request", "Other")
ticket_urgency = ("Low", "Mid", "High")
ticket_status = ("Open", "In Progress", "Completed", "Rejected")

class Ticket(Schema):
    id = fields.Int(dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    name = fields.Str(required=True)
    email = fields.Email(required=True)
    subject = fields.Str(required=True)
    created_at = fields.DateTime(dump_only=True)
    message = fields.Str(required=True)
    type = fields.Str(required=True, validate=OneOf(ticket_type))
    urgency = fields.Str(required=True, validate=OneOf(ticket_urgency))
    status = fields.Str(
        missing="Open", required=True, validate=OneOf(ticket_status)
        )


class Comment(Schema):
    id = fields.Int(dump_only=True)
    message = fields.Str(required=True)
    created_at = fields.DateTime(dump_only=True)

class User(Schema):
    email = fields.Str(required=True)
    password = fields.Str(required=True)
