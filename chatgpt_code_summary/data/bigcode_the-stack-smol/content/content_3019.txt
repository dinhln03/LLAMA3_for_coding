from flask import render_template,redirect,url_for,request,flash
from . import auth
from ..models import Group
from .forms import RegistrationForm,LoginForm
from .. import db
from flask_login import login_user,logout_user,login_required


@auth.route('/login', methods=["GET", "POST"])
def login():
    login_form = LoginForm()

    if login_form.validate_on_submit():

        group = Group.query.filter_by( name=login_form.name.data).first()

        if group is not None and group.verify_password(login_form.password.data):

            login_user(group, login_form.remember.data)

            return redirect(request.args.get('next') or url_for('main.group', id=group.id))

        flash('Invalid group name or password')

    title="Login"

    return render_template('auth/login.html', login_form=login_form, title=title)

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for("main.index"))


@auth.route('/register', methods=["GET", "POST"])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        group = Group( name=form.name.data, password=form.password.data)

        db.session.add(group)

        db.session.commit()

        return redirect(url_for('auth.login'))

    title="New Account"

    return render_template('auth/register.html', registration_form=form, title=title)




