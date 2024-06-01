import sqlite3
from .login import login

lg = login()

class account():

    def atualizarBanco(self, email, nome, senha, cidade, aniversario, sexo, visibilidade=1):
        conn = sqlite3.connect('./database/project.db', check_same_thread=False)
        c = conn.cursor()
        idLogado = lg.verLogado()
        c.execute('PRAGMA foreign_keys = ON;')
        c.execute('UPDATE user SET email_user = ?, nome_user = ?, senha_user = ?, cidade_user = ?, aniversario = ?, sexo = ?, visibilidade = ? WHERE id_User = ?', (email, nome, senha, cidade, aniversario, sexo, visibilidade, idLogado))
        conn.commit()
        return True