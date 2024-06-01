# from flask import Flask, Blueprint
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager
# import os

from flask import Flask, jsonify, request, make_response, redirect, url_for
import jwt
import datetime
import os
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import select
from flask_migrate import Migrate, migrate
from flask_cors import CORS
from sqlalchemy import inspect
from sqlalchemy import Table, Column, MetaData, Integer, Computed
from numpy import array

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretollave'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todo.db'
ABSOLUTE_PATH_TO_YOUR_FOLDER ='/home/dani/flask/static/fotosPerfil'
ABSOLUTE_PATH_TO_YOUR_PDF_FOLDER ='/home/dani/flask/static/pdf'
CORS(app)
db  = SQLAlchemy(app)
migrate = Migrate(app, db)


# Models
class Usuario(db.Model):
    nick = db.Column(db.String(20), primary_key=True)
    Nombre_de_usuario = db.Column(db.String(50))
    password = db.Column(db.String(50))
    e_mail = db.Column(db.String(50), unique=True, nullable=False)
    descripcion  = db.Column(db.String(1000))
    link  = db.Column(db.String(200))
    foto_de_perfil = db.Column(db.String(400))

class Sigue(db.Model):
    #id  = db.Column(db.Integer, primary_key=True )
    Usuario_Nicka = db.Column(db.String(20), db.ForeignKey('usuario.nick'),primary_key=True)
    Usuario_Nickb = db.Column(db.String(20), db.ForeignKey('usuario.nick'),primary_key=True)

class Chat(db.Model):

    #Column('timestamp', TIMESTAMP(timezone=False), nullable=False, default=datetime.now())
    timestamp = db.Column(db.TIMESTAMP, nullable=False,
                  server_default=db.func.now(),
                  onupdate=db.func.now())

    mensaje  = db.Column(db.String(1000))
    Usuario_Nicka = db.Column(db.String(20), db.ForeignKey('usuario.nick'),primary_key=True)
    Usuario_Nickb = db.Column(db.String(20), db.ForeignKey('usuario.nick'),primary_key=True)


class Publicacion(db.Model):

    id  = db.Column(Integer,primary_key=True)
    #id = db.Sequence('id', start=1, increment=1)
    descripcion  = db.Column(db.String(1000))
    #Column('timestamp', TIMESTAMP(timezone=False), nullable=False, default=datetime.now())
    timestamp = db.Column(db.TIMESTAMP, nullable=False,
                  server_default=db.func.now(),
                  onupdate=db.func.now())
    Usuario_Nicka = db.Column(db.String(20), db.ForeignKey('usuario.nick'))

class Propia(db.Model):

    pdf = db.Column(db.String(400))
    id = db.Column(db.String(20), db.ForeignKey('publicacion.id'),primary_key=True)


class Recomendacion(db.Model):

    link  = db.Column(db.String(200),nullable=False)
    titulo = db.Column(db.String(200),nullable=False)
    autor  = db.Column(db.String(200),nullable=False)
    id = db.Column(db.String(20), db.ForeignKey('publicacion.id'),primary_key=True)

class Tematica(db.Model):

    tema  = db.Column(db.String(50), primary_key=True )


class Notificaciones(db.Model):

    id  = db.Column(db.Integer, primary_key=True )
    fecha  = db.Column(db.Date)
    Usuario_Nicka = db.Column(db.String(20), db.ForeignKey('usuario.nick'),primary_key=True)


class Prefiere(db.Model):

    Usuario_Nicka = db.Column(db.String(20), db.ForeignKey('usuario.nick'),primary_key=True)
    tema = db.Column(db.String(50), db.ForeignKey('tematica.tema'),primary_key=True)


class Trata_pub_del_tema(db.Model):

    id = db.Column(db.Integer, db.ForeignKey('publicacion.id'),primary_key=True)
    tema = db.Column(db.String(50), db.ForeignKey('tematica.tema'),primary_key=True)

class Gusta(db.Model):

    id = db.Column(db.Integer, db.ForeignKey('publicacion.id'),primary_key=True)
    Usuario_Nicka = db.Column(db.String(20), db.ForeignKey('usuario.nick'),primary_key=True)


class Comenta(db.Model):

    id = db.Column(db.Integer, db.ForeignKey('publicacion.id'),primary_key=True)
    Usuario_Nicka = db.Column(db.String(20), db.ForeignKey('usuario.nick'),primary_key=True)
    comentario  = db.Column(db.String(1000))

class Guarda(db.Model):

    id = db.Column(db.Integer, db.ForeignKey('publicacion.id'),primary_key=True)
    Usuario_Nicka = db.Column(db.String(20), db.ForeignKey('usuario.nick'),primary_key=True)

class Trata(db.Model):

    id_publi = db.Column(db.Integer, db.ForeignKey('publicacion.id'),primary_key=True)
    id_notif = db.Column(db.String(20), db.ForeignKey('notificaciones.id'),primary_key=True)


class Genera(db.Model):

    id = db.Column(db.Integer, db.ForeignKey('publicacion.id'),primary_key=True)
    Usuario_Nicka = db.Column(db.String(20), db.ForeignKey('usuario.nick'),primary_key=True)




def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        #token = request.args.get('token') #http://127.0.0.1:5000/route?token=djsnvidnoffofn
        #data = request.get_json()
        token = request.headers['token']
        #token = data['token']
        if not token:
            return jsonify({'error': 'Token no existe'}), 403

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = Usuario.query.filter_by(nick=data['nick']).first()
            current_user = data['nick']
        except:
            return jsonify({'error': 'Token no valido'}), 403

        return f(current_user,*args, **kwargs)
    return decorated


@app.route('/unprotected')
def unprotected():
    return jsonify({'message': 'Puede entrar tol mundo'})

@app.route('/protected')
@token_required
def protected(current_user):
    print(current_user)
    return jsonify({'message': 'Puedes entrar si puedes'})

# Ruta para el login



@app.route('/register', methods=['POST'])
def add_data():
    data= request.get_json()
    #nick = request.form.get("nick")
    #password = request.form.get("password")
    #e_mail = request.form.get("e_mail")


    user = Usuario.query.filter_by(e_mail=data['e_mail']).first()
    nick = Usuario.query.filter_by(nick=data['nick']).first()
    if user: # si esto devuelve algo entonces el email existe
        return jsonify({'error': 'Existe correo'}) #json diciendo error existe email
    if nick:
        return jsonify({'error': 'Existe nick'})
    #if (check_email(e_mail) == True and check_password(data['password']) == True ):
    register = Usuario(nick=data['nick'],password=generate_password_hash(data['password']), e_mail=data['e_mail'],foto_de_perfil="platon.jpg")
    db.session.add(register)
    db.session.commit()


    token = jwt.encode({'nick' : data['nick'], 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)}, app.config['SECRET_KEY'])
    return jsonify({'token' : token.decode('UTF-8')})



@app.route('/login', methods=['POST'])
def login():
    # auth = request.authorization #new ESTO SI LO HACES CON AUTH

    data= request.get_json()

    if '@' in data['nickOcorreo']:
        user = Usuario.query.filter_by(e_mail=data['nickOcorreo']).first()
    else:
        user = Usuario.query.filter_by(nick=data['nickOcorreo']).first()

    if not user:
        return jsonify({'error': 'No existe ese usuario'})#error mal user
    if not check_password_hash(user.password, data['password']):
        return jsonify({'error': 'Mal contraseña'}) #error mala contraseña


    token = jwt.encode({'nick' : data['nickOcorreo'], 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=9999999)}, app.config['SECRET_KEY'])
    return jsonify({'token' : token.decode('UTF-8')})




@app.route('/editarPerfil', methods=['GET'])
@token_required
def editarPerfilget(current_user):
    s = select([Usuario.Nombre_de_usuario,  Usuario.descripcion,Usuario.link, Usuario.foto_de_perfil]).where((Usuario.nick == current_user))
    result = db.session.execute(s)

    seguidos= db.session.query(Sigue).filter(Sigue.Usuario_Nicka == current_user ).count()
    seguidores= db.session.query(Sigue).filter(Sigue.Usuario_Nickb == current_user ).count()
    nposts= db.session.query(Publicacion).filter(Publicacion.Usuario_Nicka == current_user ).count()

    tema = select([Prefiere.tema]).where((Prefiere.Usuario_Nicka == current_user))
    temas = db.session.execute(tema)
    vector = []
    for row in temas:
        vector += row
    for row in result:
        fila = {
            "nick": current_user,
            "nombre_de_usuario":row[0],
            "descripcion":row[1],
            "link":row[2],
            "foto_de_perfil": 'http://51.255.50.207:5000/display/' + row[3],
            "nsiguiendo": seguidos,
            "nseguidores": seguidores,
            "nposts": nposts,
            "tematicas": vector
            #"foto_de_perfil" :url_for('static', filename='fotosPerfil/' + row[3])
        }
    return fila

@app.route('/display/<filename>')
def foto(filename):
    return redirect(url_for('static', filename='fotosPerfil/' + filename),code = 301)


@app.route('/editarPerfil', methods=['POST'])
@token_required
def editarPerfilpost(current_user):

    data= request.get_json()
    user = Usuario.query.filter_by(nick=current_user).first()
    user.Nombre_de_usuario = data['nombre_de_usuario']
    print(data['nombre_de_usuario'])
    print(data['descripcion'])
    print(data['link'])
    print(data['tematicas'])
    user.descripcion = data['descripcion']
    user.link = data['link']
    tematicas = data['tematicas']
    for temas in tematicas:
        tema = Prefiere.query.filter_by(tema=temas).first()
        if not tema:
            tema = Prefiere(Usuario_Nicka=current_user, tema = temas)
            db.session.add(tema)
        #db.session.commit()
    #cambia_foto

    db.session.commit()

    token = jwt.encode({'nick' : current_user, 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)}, app.config['SECRET_KEY'])
    return jsonify({'token' : token.decode('UTF-8')})


@app.route('/actualizarImagen', methods=['POST'])
@token_required
def actualizarImagen(current_user):
    user = Usuario.query.filter_by(nick=current_user).first()

    if request.files['nueva_foto'] is not None: #data['cambia_foto']:

    	file = request.files['nueva_foto']
    	print(request.files['nueva_foto'])
    	filename = secure_filename(file.filename)
    	file.save(os.path.join(ABSOLUTE_PATH_TO_YOUR_FOLDER, filename))
    	user.foto_de_perfil = filename
    	db.session.commit()

    token = jwt.encode({'nick' : current_user, 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)}, app.config['SECRET_KEY'])
    return jsonify({'token' : token.decode('UTF-8')})

@app.route('/subirPost', methods=['POST'])
@token_required
def subirPost(current_user):

    data= request.get_json()

    publicacion = Publicacion(descripcion=data['descripcion'],Usuario_Nicka=current_user) #coger id
    db.session.add(publicacion)
    db.session.commit()

    tematicas = data['tematicas']
    for temas in tematicas:
        temita = Tematica.query.filter_by(tema=temas).first()
        if temita:
            nuevo = Trata_pub_del_tema(id=publicacion.id, tema = temita.tema)
            db.session.add(nuevo)
    db.session.commit()
    if (data['tipo']=="1"): # articulo
        print("xd")
        guardarPDF(request.files['pdf'], publicacion.id)
    elif(data['tipo']=="2"): # recomendacion
        recomendacion = Recomendacion(link=data['link'],titulo=data['titulo'], autor = data['autor'], id = publicacion.id)
        db.session.add(recomendacion)
        
    
    db.session.commit()
    token = jwt.encode({'nick' : current_user, 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)}, app.config['SECRET_KEY'])
    return jsonify({'token' : token.decode('UTF-8')})


def guardarPDF(pdf,_id):
    propia = Propia.query.filter_by(id=_id).first()
    if pdf is not None:
    	file = pdf
    	print(pdf)
    	filename = secure_filename(file.filename)
    	file.save(os.path.join(ABSOLUTE_PATH_TO_YOUR_PDF_FOLDER, filename))
    	propia.pdf = filename
    	db.session.add(propia)


@app.route('/getPostsPropios', methods=['GET'])
@token_required
def getPostsPropios(current_user):

    data= request.get_json()

    a = select([Usuario.Nombre_de_usuario]).where((Usuario.nick == current_user))
    resulta = db.session.execute(a)
    #s = select([Publicacion.Usuario_Nicka,  Publicacion.descripcion,Publicacion.timestamp]).where((Publicacion.Usuario_Nicka == current_user and Publicacion.id>data['id']-8 and and Publicacion.id<=data['id'])).order_by(Publicacion.id)
    
    s=select(Publicacion).where(Publicacion.Usuario_Nicka == current_user).order_by(Publicacion.id.desc())
    results = db.session.execute(s)

    
    for r in results:
        for i in range(data['id']-8,data['id']):
            a = select([Propia.id,  Propia.pdf]).where((Propia.id == r.id))
            resulta = db.session.execute(a)

            Gustas= db.session.query(Gusta).filter(Gusta.Usuario_Nicka == current_user, Gusta.id == row[1] ).count()
            Comentarios= db.session.query(Comenta).filter(Comenta.Usuario_Nicka == current_user, Comenta.id == row[1] ).count()
            Guardados= db.session.query(Guarda).filter(Guarda.Usuario_Nicka == current_user, Guarda.id == row[1] ).count()

            fila = {
                "id": r.id,
                "nick": current_user,
                "descripcion":r.descripcion,
                "timestamp":r.timestamp,
                "pdf": 'http://51.255.50.207:5000/display2/' + a.pdf,
                "nlikes": Gustas,
                "ncomentarios": Comentarios,
                "nguardados": Guardados,
                "usuario": resulta.nombre_de_usuario
            }
    
    return fila


@app.route('/display2/<filename>')
def pdf(filename):
    return redirect(url_for('static', filename='pdf/' + filename),code = 301)

@app.route('/getPostsRecomendados', methods=['GET'])
@token_required
def getPostsRecomendados(current_user):

    #data= request.get_json()

    a = select([Usuario.Nombre_de_usuario]).where((Usuario.nick == current_user))
    resultb = db.session.execute(a)
    Nombre_de_usuario = ""
    for b in resultb: 
        Nombre_de_usuario=b.Nombre_de_usuario
    #s = select([Publicacion.Usuario_Nicka,  Publicacion.descripcion,Publicacion.timestamp]).where((Publicacion.Usuario_Nicka == current_user and Publicacion.id>data['id']-8 and and Publicacion.id<=data['id'])).order_by(Publicacion.id)
    
    s = select([Publicacion]).where(Publicacion.Usuario_Nicka == current_user).order_by(Publicacion.id.desc())

    results = db.session.execute(s)
    
    # for record in results:
    #     print("\n", record)

    vector0 = array([])

    vector1 = []
    vector2 = []
    
    for r in results:
        print(str(r.id))
        vector0 += r.id
        vector1 += str(r.descripcion)
        vector2 += str(r.timestamp)
    
    # for r in results:
    #     for b in resultb: 
            # a = select([Recomendacion.id,  Recomendacion.link,Recomendacion.titulo,Recomendacion.autor]).where((Recomendacion.id == r.id))
            # resulta = db.session.execute(a)
            # for a in resultaa:
            #     Gustas= db.session.query(Gusta).filter(Gusta.Usuario_Nicka == current_user, Gusta.id == r.id ).count()
            #     Comentarios= db.session.query(Comenta).filter(Comenta.Usuario_Nicka == current_user, Comenta.id == r.id ).count()
            #     Guardados= db.session.query(Guarda).filter(Guarda.Usuario_Nicka == current_user, Guarda.id == r.id ).count()

                
    print(vector0)
    fila = {
                "id": vector0,
                #"link": a.link,
                #"titulo": a.titulo,
                #"autor": a.autor,
                "nick": current_user,
                "descripcion": vector1,
                "timestamp": vector2,
                #"nlikes": Gustas,
                #"ncomentarios": Comentarios,
                #"nguardados": Guardados,
                "usuario": Nombre_de_usuario
                }
        
    return fila

def check_email(email):

    regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'

    if(re.search(regex,email)):
        return True
    else:
        return False

# Contraseñas de entre 8 y 32 carácteres.

def check_password(password):

    regex = '^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[*.!@$%^&(){}[]:;<>,.?/~_+-=|\]).{8,32}$'

    if(re.search(regex,password)):
        return True
    else:
        return False



if __name__ == '__main__':
    app.run(debug=True)























