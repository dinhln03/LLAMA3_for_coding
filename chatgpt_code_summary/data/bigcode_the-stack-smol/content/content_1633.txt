from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
from flask_mail import Mail, Message
import bcrypt
import re
from validate_email import validate_email
from validate_docbr import CPF
from sqlalchemy.ext.declarative import declarative_base
from flask_marshmallow import Marshmallow
from flask_cors import CORS, cross_origin
from marshmallow import fields


Base = declarative_base()

app = Flask(__name__)
mail= Mail(app)
CORS(app, support_credentials=True)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = 'thisissecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///banco.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = ''
app.config['MAIL_PASSWORD'] = ''
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

db = SQLAlchemy(app)
ma = Marshmallow(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True)
    name = db.Column(db.String(50))
    cpf = db.Column(db.String(11))
    birthdate = db.Column(db.String(10))
    gender = db.Column(db.String(1))
    phone = db.Column(db.String(11))
    email = db.Column(db.String(50))
    password = db.Column(db.String(80))
    passwordResetToken = db.Column(db.String(250))
    passwordResetExpires = db.Column(db.String(100))

class Product(db.Model, Base):
    __tablename__ = 'products'
    product_id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(250))
    price = db.Column(db.Float)
    installments = db.Column(db.Integer)
    sizes = db.Column(db.String(50))
    availableSizes = db.Column(db.String(50))
    gender = db.Column(db.String(1))
    material = db.Column(db.String(50))
    color = db.Column(db.String(50))
    brand = db.Column(db.String(50))
    carts = db.relationship('Cart',secondary='cart_products')

class Image(db.Model, Base):
    __tablename__ = 'products_imgs'
    img_id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(300))
    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id'))
    product = db.relationship('Product', backref='images')

class Cart(db.Model, Base):
    __tablename__ = 'cart'
    cart_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    total_amount = db.Column(db.Float)
    create_dttm = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    user = db.relationship('User', backref='images')
    products = db.relationship('Product', secondary = 'cart_products')

class CP(db.Model, Base):
    __tablename__ = 'cart_products'
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id'))
    cart_id = db.Column(db.Integer, db.ForeignKey('cart.cart_id'))
    quantity = db.Column(db.Integer)
    size = db.Column(db.String(5))
    product = db.relationship(Product, backref=db.backref("cart_products", cascade="all, delete-orphan"))
    cart = db.relationship(Cart, backref=db.backref("cart_products", cascade="all, delete-orphan"))
    
class ImageSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Image
        include_fk = True

class ProductSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Product
    images = fields.Nested(ImageSchema, many=True, only=['url'])



def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        
        if not token:
            return jsonify({'message' : 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = User.query.filter_by(public_id=data['public_id']).first()
        except:
            return jsonify({'message' : 'Token is invalid!'}), 401

        return f(current_user, *args, **kwargs)
    
    return decorated


@app.route('/user/<public_id>', methods=['GET'])
@cross_origin(supports_credentials=True)
@token_required
def get_one_user(current_user, public_id):

    user = User.query.filter_by(public_id=public_id).first()

    if not user:
        return jsonify({'message' : 'Usuário não encontrado!'}), 400

    user_data = {}
    user_data['public_id'] = user.public_id
    user_data['name'] = user.name
    user_data['cpf'] = user.cpf
    user_data['birthdate'] = user.birthdate
    user_data['gender'] = user.gender
    user_data['phone'] = user.phone
    user_data['email'] = user.email

    return jsonify({'user' : user_data}), 200


@app.route('/users', methods=['POST'])
@cross_origin(supports_credentials=True)
def create_user():
    cpf = CPF()

    data = request.get_json()

    if not all(x.isalpha() or x.isspace() for x in str(data['name'])) or len(str(data['name'])) < 3 or len(str(data['name'])) > 100:
        return jsonify({'message' : 'Nome inválido!'}), 400
    elif not cpf.validate(str(data['cpf'])):
        return jsonify({'message' : 'CPF inválido!'}), 400
    elif datetime.date.today().year - datetime.datetime.strptime(str(data['birthdate']), "%d/%m/%Y").year < 18:
        return jsonify({'message' : 'Usuário menor de idade!'}), 400
    elif str(data['gender']) != "M" and str(data['gender']) != "F":
        return jsonify({'message' : 'Gênero inválido!'}), 400
    elif not str(data['phone']).isdigit() or len(str(data['phone'])) < 10:
        return jsonify({'message' : 'Telefone inválido!'}), 400
    elif not validate_email(str(data['email'])):
        return jsonify({'message' : 'Email inválido!'}), 400
    elif len(str(data['password'])) < 8 or len(str(data['password'])) > 20:
        return jsonify({'message' : 'Senha inválida!'}), 400

    prospect_cpf = User.query.filter_by(cpf=data['cpf']).first()
    prospect_email = User.query.filter_by(email=data['email']).first()
    
    if prospect_cpf:
        return jsonify({'message' : 'CPF já cadastrado!'}), 400
    elif prospect_email:
        return jsonify({'message' : 'Email já cadastrado!'}), 400

    hashed_password = generate_password_hash(data['password'], method='sha256')
    new_user = User(public_id=str(uuid.uuid4()), name=data['name'], cpf=data['cpf'], birthdate=data['birthdate'],
                gender=data['gender'], phone=data['phone'], email=data['email'], password=hashed_password, passwordResetToken=None, passwordResetExpires=None)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message' : 'Usuário cadastrado com sucesso!'}), 200


@app.route('/users/<public_id>', methods=['DELETE'])
@cross_origin(supports_credentials=True)
@token_required
def delete_user(current_user, public_id):

    user = User.query.filter_by(public_id=public_id).first()

    if not user:
        return jsonify({'message' : 'Usuário não encontrado'}), 400

    db.session.delete(user)
    db.session.commit()

    return jsonify({'message' : 'Usuário apagado com sucesso!'}), 200


@app.route('/login', methods=['POST'])
@cross_origin(supports_credentials=True)
def login():
    auth = request.get_json()

    if not auth or not auth['email'] or not auth['password']:
        return jsonify({'message' : 'Email ou senha não foram preenchidos!'}), 401

    user = User.query.filter_by(email=auth['email']).first()

    if not user:
        return jsonify({'message' : 'Email não existe!'}), 401

    if check_password_hash(user.password, auth['password']):
        token = jwt.encode({'public_id' : user.public_id, 'exp' : datetime.datetime.utcnow() + datetime.timedelta(minutes=30)}, app.config['SECRET_KEY'])

        return jsonify({'token' : token.decode('UTF-8'), 'id' : user.public_id, 'name' : user.name, 'email' : user.email}), 200

    return jsonify({'message' : 'Senha incorreta'}), 401


@app.route("/forgot-password", methods=['POST'])
@cross_origin(supports_credentials=True)
def send_email():
    data = request.get_json()

    user = User.query.filter_by(email=data['email']).first()

    if not user:
        return jsonify({'message' : "Email não encontrado!"}), 400

    password = str(user.email).encode('UTF-8')
    passToken = bcrypt.hashpw(password, bcrypt.gensalt())
    passToken = re.sub('\W+','', str(passToken))
    passExpires = str(datetime.datetime.utcnow() + datetime.timedelta(minutes=15))

    user.passwordResetToken = passToken
    user.passwordResetExpires = passExpires
    db.session.commit()

    msg = Message('Recuperação de senha - Gama Sports', sender = app.config['MAIL_USERNAME'], recipients = [user.email])
    msg.body = "Olá " + str(user.email) + ", \n\n" + "Acesse o link a seguir para trocar sua senha: \n\n" + "http://localhost:4200/users/recover-password?token=" + str(passToken)
    mail.send(msg)

    return jsonify({'message' : "Email disparado!"}), 200


@app.route("/reset-password", methods=['POST'])
@cross_origin(supports_credentials=True)
def change_password():
    data = request.get_json()

    user = User.query.filter_by(passwordResetToken=str(data['token'])).first()

    if not user:
        return jsonify({'message' : "Token inválido!"}), 400

    date_time_exp = datetime.datetime.strptime(user.passwordResetExpires, '%Y-%m-%d %H:%M:%S.%f')

    if datetime.datetime.utcnow() > date_time_exp:
        return jsonify({'message' : "Token expirado, gere um novo!"}), 400

    if len(str(data['password'])) < 8 or len(str(data['password'])) > 20:
        return jsonify({'message' : 'Senha inválida!'}), 400

    hashed_newpassword = generate_password_hash(data['password'], method='sha256')

    user.password = hashed_newpassword
    user.passwordResetToken = None
    user.passwordResetExpires = None
    db.session.commit()

    return jsonify({'message' : "Senha trocada com sucesso!"}), 200


@app.route('/products', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_all_products():

    search = request.args.get("search", None)

    if not search:
        products = Product.query.all()
    else:
        search = "%{}%".format(search)
        products = Product.query.filter(Product.description.like(search)).all()
        if not products:
            return jsonify([]), 200

    product_schema = ProductSchema(many=True)
    output = product_schema.dump(products)
        
    return jsonify(output), 200


@app.route('/products/<product_id>', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_product(product_id):

    product = Product.query.filter_by(product_id=product_id).first()

    if not product:
        return jsonify({'message' : 'Produto não encontrado!'}), 400

    product_schema = ProductSchema()
    output = product_schema.dump(product)
        
    return jsonify(output), 200


@app.route('/cart', methods=['POST'])
@cross_origin(supports_credentials=True)
@token_required
def create_cart(current_user):

    data = request.get_json()

    cart = Cart(total_amount=data['total'], user_id=data['clientId'])
    db.session.add(cart)
    db.session.commit()

    if not cart:
        return jsonify({'message' : 'Problema na inclusão do carrinho'}), 400

    for product in data['products']:
        if not product:
            return jsonify({'message' : 'Problema na inclusão do produto'}), 400
        add_product = CP(product_id=product['id'], cart_id=cart.cart_id, quantity=product['quantity'], size=product['size'])
        db.session.add(add_product)
    db.session.commit()
    
    return jsonify({'message' : 'Carrinho salvo com sucesso!'}), 200


if __name__ == '__main__':
    app.run(debug=True)