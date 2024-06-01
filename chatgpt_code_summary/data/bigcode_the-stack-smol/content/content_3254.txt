#####################
# IMPORT DEPENDENCIES
######################

# flask (server)
from flask import(
    Flask, 
    render_template, 
    jsonify, 
    request,
    redirect)

#######################
# FLASK SET-UP
#######################
app = Flask(__name__)

#######################
# FLASK ROUTES
#######################

@app.route("/")
def index():
    return render_template("index.html")

# @app.route("/outcomes")
# def charts():
#     return render_template("outcomes.html")

if __name__ == "__main__":
    app.run(debug = True)