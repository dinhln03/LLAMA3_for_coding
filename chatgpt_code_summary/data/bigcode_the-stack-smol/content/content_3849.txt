from flask import render_template, url_for, request, flash, redirect, make_response
import email
from app import app
from werkzeug.utils import secure_filename
from app.predict_email import Prediction
import tempfile

predict_email = Prediction()

def parse_email(email_raw):
    parser = email.parser.BytesParser()
    email_parsed = parser.parse(email_raw)
    return email_parsed
    

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        email_raw = request.files["email_raw"]
        
        if email_raw.filename != "":
            temp_name = next(tempfile._get_candidate_names())
            with open(f"./app/data/uploads/{temp_name}.eml", "wb") as f:
                f.write(email_raw.read())

            spam,prediction = predict_email.predict_emails([f"./app/data/uploads/{temp_name}.eml"])

            # email_parsed = parse_email(email_raw)
            # print(email["subject"])
            # Features = prepData(textData)
            # prediction = int((np.asscalar(loaded_model.predict(Features))) * 100)

            if spam:
                page = "spam.html"
                score = int(round(prediction[0][1]*100))
            else:
                page = "ham.html"
                score = int(round(prediction[0][0]*100))

            r = make_response(render_template(page, prediction=score))
            r.headers.add('Access-Control-Allow-Origin', '*')
            r.headers.add('Access-Control-Expose-Headers', 'Content-Disposition')
            return r

        else:
            return render_template("home.html")

    else:
        return render_template("home.html")


@app.route("/predict2")
def predict2():
    return render_template("ham.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     df = pd.read_csv("spam.csv", encoding="latin-1")
#     df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
#     # Features and Labels
#     df["label"] = df["class"].map({"ham": 0, "spam": 1})
#     X = df["message"]
#     y = df["label"]

#     # Extract Feature With CountVectorizer
#     cv = CountVectorizer()
#     X = cv.fit_transform(X)  # Fit the Data
#     from sklearn.model_selection import train_test_split

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.33, random_state=42
#     )
#     # Naive Bayes Classifier
#     from sklearn.naive_bayes import MultinomialNB

#     clf = MultinomialNB()
#     clf.fit(X_train, y_train)
#     clf.score(X_test, y_test)
#     # Alternative Usage of Saved Model
#     # joblib.dump(clf, 'NB_spam_model.pkl')
#     # NB_spam_model = open('NB_spam_model.pkl','rb')
#     # clf = joblib.load(NB_spam_model)
#     if request.method == "POST":
#         message = request.form["message"]
#         data = [message]
#         vect = cv.transform(data).toarray()
#         my_prediction = clf.predict(vect)
#     return render_template("result.html", prediction=my_prediction)
