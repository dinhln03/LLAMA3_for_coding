from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SubmitField, ValidationError, BooleanField, TextAreaField,SelectField
from wtforms.validators import Required,Email,EqualTo
from ..models import User

class CommentForm(FlaskForm):
    comment = TextAreaField('Your comment:', validators=[Required()])
    submit = SubmitField('Comment')

pitch_category = [('Pickup Lines', 'Pickup Lines'), ('Interview Pitch', 'Inteview Pitch'), ('Product Pitch', 'Product Pitch'), ('Promo Pitch', 'Promo Pitch')]

class PitchForm(FlaskForm):
    category = SelectField('Category', choices=pitch_category)
    pitch = TextAreaField('Your pitch:', validators=[Required()])
    submit = SubmitField('Submit Pitch')