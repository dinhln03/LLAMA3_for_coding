from flask import Flask
from flask import render_template

app = Flask("Boostrap_Demo")


@app.route("/")
def start():
    name = "Fabian"
    cards = [
        {"titel": "Card 0", "inhalt": "Blubber"},
        {"titel": "Card 1", "inhalt": "Bla"},
        {"titel": "Card 2", "inhalt": "Käsekuchen"},
        {"titel": "Card 2", "inhalt": "Sülze"}
    ]
    return render_template("start.html", name=name, cards=cards)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
