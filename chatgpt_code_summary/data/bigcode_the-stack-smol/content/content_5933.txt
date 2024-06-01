from flask import Flask, render_template, request, flash, url_for
from flask_mail import Message, Mail
import json
from typing import Dict, List
from pathlib import Path

from forms import ContactForm
from development_config import Config
"""
This file launches the application.
"""

# init application
app = Flask(__name__)

# add secretkey, mail and debug configurations
app.config.from_object(Config)

# attaching mail to the flask app
mail = Mail(app)


def read_json(json_file: str, debug=False) -> List[Dict]:
    """
    reads the json files, and formats the description that
    is associated with each of the json dictionaries that are read in.

    :param json_file: json file to parse from
    :param debug: if set to true, will print the json dictionaries as
    they are read in
    :return: list of all of the json dictionaries
    """

    # parsing json file
    with open(json_file, "r") as json_desc:
        # read json file
        project_list: List[Dict] = json.load(json_desc)

    # formats the description data which I stored in a json list
    for project in project_list:
        project['description'] = " ".join(project['description'])
        if debug:
            print(project)

    return project_list


@app.route("/")
@app.route("/home")
def home_page():
    return render_template("home.html", title="home")


@app.route("/portfolio")
def portfolio():

    # json file to parse
    json_file = "static/json/projects.json"

    project_list = read_json(json_file)

    # grouping portfolio into two's
    project_groups = [[project_list[i*2], project_list[i*2+1]]
                      for i in range(len(project_list) // 2)]

    # getting the last project
    project_singles = False
    if len(project_list) % 2 != 0:
        project_singles = project_list[-1:]

    return render_template("portfolio.html",
                           title="portfolio",
                           project_groups=project_groups,
                           project_singles=project_singles)


@app.route("/talks")
def talks():

    # json file to parse
    json_file = "static/json/talks.json"

    # parsed json results
    project_list = read_json(json_file)

    return render_template("talks.html",
                           project_list=project_list,
                           title="talks")


@app.route("/contact", methods=['GET', 'POST'])
def contact():

    # although I am recreating this form object for every call
    # - it's state seems to persist...
    form = ContactForm()

    if request.method == 'POST':
        if form.validate() is False:
            flash("All fields are required", "flash")
            return render_template("contact.html", form=form)
        else:
            msg = Message(form.subject.data,
                          sender='jimmy.shaddix2.0@gmail.com',
                          recipients=['jimmy.shaddix2.0@gmail.com'])
            msg.body = """
                  From: {} <{}>
                  {}
                  """.format(form.name.data, form.email.data, form.message.data)
            mail.send(msg)
            return render_template('contact.html', success=True)
    elif request.method == 'GET':
        return render_template("contact.html", form=form, title="email")


if __name__ == "__main__":
    app.run()
