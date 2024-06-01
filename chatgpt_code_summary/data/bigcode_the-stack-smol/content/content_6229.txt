from flask import Flask, render_template, url_for, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
db = SQLAlchemy(app)

app.config['SQLALCHEMY_TRACK MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30))
    password = db.Column(db.String(30))


@app.route('/', methods=['POST', 'GET'])
def login():
    username = request.form['username']
    password = request.form['password']

    db.session.add(username)
    db.session.add(password)
    db.session.commit()
    return render_template("index.html")

@app.route('/secret')
def secret():
    return render_template("secret.html")

if __name__ == "__main__":
    app.run(debug=True)