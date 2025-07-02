from flask import Flask,request,jsonify
import joblib
import pandas as pd

app=Flask(__name__)

model=joblib.load('turbine_health_model.pkl')

@app.route('/')
def home():
    return "Turbine prediction api is running"

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data=request.get_json()
        input=pd.DataFrame([data])
        prediction=model.predict(input)
        return jsonify({'Prediction':prediction.tolist()})
    except Exception as e:
        return jsonify({'Error':str(e)}),400
    
if __name__ == '__main__':
    app.run(debug=True)

