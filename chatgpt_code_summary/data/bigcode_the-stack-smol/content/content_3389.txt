from markovp import Markov
from src.forms import Markov_Form
from flask import Flask, render_template, request, redirect, url_for, Blueprint, make_response

home = Blueprint("home", __name__)

@home.route("/")
def index():
#{
    form = Markov_Form()
    return render_template('form.html', form = form)
#}

# The submission page.
@home.route("/submit", methods=["GET"])
def submit():
#{
    if request.method == "GET":
    #{
        if request.args.get('submit_button'):
        #{    
            # Get form values.
            # http://stackoverflow.com/a/20341272/5415895
            text = request.args.get("input_text")
            # We have to cast text as a string, otherwise C++ complains.
            mark = Markov(str(text), 1)
            output = mark.generate()

            return render_template("output.html", input = str(text), output = output)
        #}
        else:
        #{
            # Make sure nobody can access the submit path without submitting.
            return redirect(url_for('index'))
        #}
    #}
    else:
    #{
        return redirect(url_for('index'))
    #}
#}