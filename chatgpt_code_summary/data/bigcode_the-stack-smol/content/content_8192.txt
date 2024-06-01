from crypt import methods
from unicodedata import name
from flask import Flask, render_template, request, session, logging, url_for, redirect, flash
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from passlib.hash import sha256_crypt
from flask_login import login_user

engine = create_engine("postgresql+psycopg2://moringa:1234@localhost/signup")

db = scoped_session(sessionmaker(bind=engine))
app = Flask(__name__)
app.secret_key = "1234code"


@app.route('/')
def landing():
    return render_template('landing.html')


#login page form
@app.route('/signin', methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        usernamedata = db.execute(
            "SELECT username FROM users WHERE username:=username", {
                "username": username
            }).fetchone()
        passworddata = db.execute(
            "SELECT password FROM users WHERE password:=password", {
                "password": password
            }).fetchone()

        if usernamedata is None:
            flash("NO username", "danger")
            return render_template("signin.html")

        else:
            for password_data in passworddata:
                if sha256_crypt.verify(password, password_data):

                    flash("You are logged", "success")
                    return redirect(url_for('profile'))
                else:
                    flash("Incorrect password", "danger")
                    return render_template('signin.html')

    return render_template('signin.html')


#route for photo
@app.route('/photo')
def photo():
    return render_template("photo.html")


#route for profile
@app.route('/profile')
def profile():
    return render_template('profile.html')


#register form functions,route for signup page
@app.route("/signup", methods=["POST", "GET"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        username = request.form.get("username")
        password = request.form.get("password")
        confirm = request.form.get("confirm")
        secure_password = sha256_crypt.encrypt(str(password))

        if password == confirm:
            db.execute(
                "INSERT INTO users(name,username,password) VALUES(:name,:username,:password)",
                {
                    "name": name,
                    "username": username,
                    "password": secure_password
                })
            db.commit()
            flash("You are registered and can login", "success")
            return redirect(url_for('signin'))

        else:
            flash("password did not match", "danger")
            return render_template('signup.html')

    return render_template('signup.html')


#route for contact
@app.route('/contact')
def contact():
    return render_template('contact.html')


#about us route
@app.route('/about')
def about():
    return render_template('about.html')


#route for logout
@app.route('/logout')
def logout():
    return redirect(url_for(''))


#route for social
@app.route('/social', methods=["POST", "GET"])
def social():

    return render_template("social.html")


if __name__ == "__main__":
    app.run(debug=True)