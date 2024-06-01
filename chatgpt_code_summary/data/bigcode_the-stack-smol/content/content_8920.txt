from common.database import db
from flask_restful import fields
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.sql import func


contato_campos = {
    'id': fields.Integer(attribute='id'),
    'nome': fields.String(attribute='nome'),
    'email': fields.String(attribute='email'),
    'telefone': fields.String(attribute='telefone'),
    'descricao': fields.String(attribute='descricao'),
    'isAtendido': fields.Boolean(attribute='is_atendido')
}


'''
    Classe Contato.
'''
class ContatoModel(db.Model):
    __tablename__ = 'tb_contato'

    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(255))
    email = db.Column(db.String(255))
    telefone = db.Column(db.String(13))
    descricao = db.Column(db.Text())
    is_atendido = db.Column(db.Boolean, default=False)
    dt_insercao = db.Column(db.DateTime, default=func.current_timestamp())
    
    
    def __init__(self, nome, email, telefone, descricao, is_atendido):
        self.nome = nome
        self.email = email
        self.telefone = telefone
        self.descricao = descricao
        self.is_atendido = is_atendido

    def __str__(self):
        return '<Contato %r>'%(self.nome)