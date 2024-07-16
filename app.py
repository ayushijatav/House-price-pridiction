from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('pipelinemodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    area = float(request.form['area'])
    bhk = int(request.form['bhk'])
    bathroom = int(request.form['bathroom'])
    furnishing = request.form['furnishing']
    locality = request.form['locality']
    parking = int(request.form['parking'])
    status = request.form['status']
    transaction = request.form['transaction']
    house_type = request.form['type']
    sqft = float(request.form['sqft'])

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'Area': [area],
        'BHK': [bhk],
        'Bathroom': [bathroom],
        'Furnishing': [furnishing],
        'Locality': [locality],
        'Parking': [parking],
        'Status': [status],
        'Transaction': [transaction],
        'Type': [house_type],
        'Per_Sqft': [sqft]
    })

    # Make prediction
    predicted_price = model.predict(input_data)

    # Return the predicted price
    return jsonify({'predicted_price': predicted_price[0]})

if __name__ == '__main__':
    app.run(debug=True)
