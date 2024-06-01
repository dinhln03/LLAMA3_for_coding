from flask import render_template, flash, redirect, url_for, request
from flask.views import MethodView

from app.middleware import auth
from app.models.user import User
from app.validators.register_form import RegisterForm
from app.services import avatar_service


class RegisterController(MethodView):
  @auth.optional
  def get(self):
    """
    Show register form

    Returns:
      Register template with form
    """
    return render_template('auth/register.html', form=RegisterForm())

  @auth.optional
  def post(self):
    """
    Handle the POST request and sign up the user if form validation passes

    Returns:
      A redirect or a template with the validation errors
    """
    form = RegisterForm()

    if form.validate_on_submit():
      form.validate_username(form.username)

      avatar = 'no-image.png'

      if 'avatar' in request.files and request.files['avatar']:
        avatar = avatar_service.save(form.avatar.data)

      User.create(form.username.data, form.password.data, avatar)

      flash('Your account has been created. You may now login.', 'info')

      return redirect(url_for('login'))

    return render_template('auth/register.html', form=form)
