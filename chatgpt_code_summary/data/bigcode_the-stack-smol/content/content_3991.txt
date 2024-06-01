import datetime
from app import db


class BucketList(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), unique=True)
    description = db.Column(db.Text, nullable=True)
    interests = db.Column(db.String(120), nullable=True)
    date_created = db.Column(db.DateTime, default=datetime.datetime.utcnow())
    date_modified = db.Column(db.DateTime)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    items = db.relationship('Item', backref='bucket_list_items', lazy='dynamic')

    def __repr__(self):
        return "<Bucketlist {}>".format(self.name)


class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), unique=True)
    description = db.Column(db.Text)
    status = db.Column(db.Text)
    date_accomplished = db.Column(db.DateTime)
    date_created = db.Column(db.DateTime, default=datetime.datetime.utcnow())
    date_modified = db.Column(db.DateTime)
    bucketlists = db.Column(db.Integer, db.ForeignKey('bucket_list.id'), nullable=False)

    def __repr__(self):
        return "<Items {}>".format(self.name)
