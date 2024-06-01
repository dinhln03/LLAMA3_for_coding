from . import db


class Account(db.Model):
    __tablename__ = 'account'
    account_id = db.Column(db.Integer, primary_key=True)
    account_name = db.Column(db.String(16), unique=True)
    account_pwd = db.Column(db.String(16), unique=True)
    account_nick = db.Column(db.String(16), unique=True)
    account_email = db.Column(db.String(320), unique=True)

    def __repr__(self):
        return '<Account %r>' % self.account_name

class Doc(db.Model):
    __tablename__ = 'doc'
    doc_id = db.Column(db.Integer, primary_key=True)
    doc_name = db.Column(db.String(16), unique=True)
    account_id = db.Column(db.Integer, unique=True)

    def __repr__(self):
        return '<Doc %r>' % self.doc_name

class DataSrc(db.Model):
    __tablename__ = 'datasrc'
    data_id = db.Column(db.Integer, primary_key=True)
    data_name = db.Column(db.String(16), unique=True)
    data_type = db.Column(db.String(16), unique=True)
    have_data = db.Column(db.Integer, unique=True)
    account_id = db.Column(db.Integer, unique=True)

    def __repr__(self):
        return '<DataSrc %r>' % self.data_name