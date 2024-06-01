import sys

from flask import Flask, jsonify, request, url_for
from flask_login import LoginManager, login_required, current_user
from marshmallow import ValidationError
from slugify import slugify

from entity import User, db
from model import user_schema, ma, users_schema

login_manager = LoginManager()

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///../resources/user.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db.init_app(app)
ma.init_app(app)
login_manager.init_app(app)


@app.route('/v1/user/<int:id>')
def get_user(id):
    user = User.query.get_or_404(id)
    return user_schema.jsonify(user)


@app.route('/v1/user', methods=['POST'])
def create_user():
    try:
        user = User.query.filter(User.user_name == request.form.get('user_name')).first()
        if user and user.user_name:
            raise Exception('User exist!')
        user = user_schema.load(request.form)
    except ValueError as errors:
        resp = jsonify(errors.messages)
        resp.status_code = 400
        return resp

    user.user_name = slugify(request.form.get('user_name'))
    db.session.add(user)
    db.session.commit()

    location = url_for("get_user", id=user.id)
    resp = jsonify({'message': 'created'})
    resp.status_code = 201
    resp.headers['location'] = location

    return resp


@app.route('/v1/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return users_schema.jsonify(users)


@app.route('/v1/user/<int:id>', methods=['PUT'])
def edit_user(id):
    user = User.query.get_or_404(id)
    try:
        user = user_schema.load(request.form, instance=user)
    except ValidationError as errors:
        resp = jsonify(errors.messages)
        resp.status_code = 400
        return resp

    user.user_name = slugify(user.user_name)
    db.session.add(user)
    db.session.commit()

    location = url_for("get_user", id=user.id)
    resp = jsonify({'message': 'updated'})
    resp.status_code = 201
    resp.headers['location'] = location

    return resp


@app.route('/v1/user/<int:id>', methods=['DELETE'])
def delete_user(id):
    user = User.query.get_or_404(id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "deleted"})


@app.errorhandler(404)
def page_not_found(error):
    resp = jsonify({"error": "not found"})
    resp.status_code = 404
    return resp


@app.route('/profile')
@login_required
def user_profile():
    return jsonify(current_user)


@app.route('/whoami')
def who_am_i():
    if current_user.is_authenticated:
        name = current_user.name
    else:
        name = 'Anonymous'
    return jsonify({'name': name})


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


@login_manager.request_loader
def load_user_from_request(request):
    api_key = request.headers.get('Authorization')
    if not api_key:
        return None
    return User.query.filter_by(api_key=api_key).first()


if __name__ == "__main__":
    if "createdb" in sys.argv:
        with app.app_context():
            db.create_all()
        print("Database created!")

    elif "seeddb" in sys.argv:
        with app.app_context():
            p1 = User(address="205 nguyen duy trinh", name="hoang", user_name="hoang",
                      image_url="http://example.com/rover.jpg", api_key="abc123")
            db.session.add(p1)
            p2 = User(address="truong quang trach", name="tuan", user_name="nguyen",
                      image_url="http://example.com/spot.jpg", api_key="abc345")
            db.session.add(p2)
            db.session.commit()
        print("Database seeded!")

    else:
        app.run(debug=True)
