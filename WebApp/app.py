# Path: webapp/app.py
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler


app = Flask(__name__,static_url_path='/static')
model = os.path.join(os.path.dirname(__file__), "stat_models", "classification_model.pkl")
#scaler = pickle.load(open('scaler.pkl', 'rb'))

# Standardize the data
scaler = StandardScaler()
#scaler.fit(X)
#X = scaler.transform(X)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)    
    prediction = model.predict(final_features)
    y_probabilities_test = model.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    print("Final Features",final_features)
    print("Prediction:",prediction)
    output = round(prediction[0], 2)
    y_prob=round(y_prob_success[0], 3)
    print(output)

    if output == 1:
        return render_template('layout.html', prediction_text='The transaction is fraudulent')
    else:
        return render_template('layout.html', prediction_text='The transaction is not fraudulent')

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)































#from flask import Flask, render_template, request, redirect, url_for, flash
#
#app = Flask(__name__)
#
#@app.route('/')
#def index():
#    return render_template('index.html')
#    
#
## Get a predict route for the form
#@app.route('/predict', methods=['POST'])
#def predict():
#    if request.method == 'POST':
#        # Get the data from the POST request.
#        data = request.form.to_dict()
#        print(data)
#        return redirect(url_for('index'))
#
#
#@app.route('/login', methods=['GET', 'POST'])
#def login():
#    if request.method == 'POST':
#        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
#            error = 'Invalid Credentials. Please try again.'
#        else:
#            return redirect(url_for('index'))
#    return render_template('login.html')
#
#if __name__ == '__main__':
#    app.run(debug=True)
#