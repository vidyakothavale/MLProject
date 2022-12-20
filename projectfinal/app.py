import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__,template_folder='template')
model = pickle.load(open("RandomForest.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("index.html")
@flask_app.route("/predict", methods = ["POST"])
def predict():
    
    float_features = [float(x)  for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    if prediction[0] < 0 and prediction[0] > 30:
        prediction="Air is not polluted.."
    else:
       prediction= "Air is polluted..."
         
    return render_template('index.html',prediction=prediction)
    #return render_template("index.html", prediction_text = "The Absolute Humidity is {}".format(prediction)  )   
if __name__ == "__main__":
    flask_app.run(debug=True,port=8080)
    