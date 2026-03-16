from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

tabella = pd.read_csv("infezioni_nosocomiali.csv")
le = LabelEncoder()
tabella['sesso'] = le.fit_transform(tabella['sesso'])
X = tabella.drop("infezione", axis=1)
y = tabella["infezione"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sesso_num = 1 if data["sesso"] == "M" else 0
    paziente = [[
        int(data["eta"]),
        sesso_num,
        int(data["giorni_ospedale"]),
        int(data["catetere_venoso"]),
        int(data["ventilazione"]),
        int(data["antibiotici"]),
        int(data["chirurgia"]),
        int(data["immunodeficienza"]),
    ]]
    predizione = model.predict(paziente)[0]
    proba = model.predict_proba(paziente)[0]
    return jsonify({
        "rischio": int(predizione),
        "probabilita_rischio": round(float(proba[1]) * 100, 1),
        "probabilita_no_rischio": round(float(proba[0]) * 100, 1),
    })


if __name__ == "__main__":
    app.run(debug=True)
