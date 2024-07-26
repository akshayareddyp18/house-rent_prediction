from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
house_data = pd.read_csv('hyd.csv')
new_data = house_data.drop(['amenities', 'locality', 'balconies', 'lift', 'active', 'loanAvailable', 'location', 
                            'ownerName', 'parkingDesc', 'propertyTitle', 'propertyType', 'combineDescription', 
                            'completeStreetName', 'facing', 'facingDesc', 'furnishingDesc', 'gym', 'id', 
                            'isMaintenance', 'weight', 'waterSupply', 'swimmingPool', 'shortUrl', 
                            'sharedAccomodation', 'reactivationSource'], axis=1)
new_data2=new_data.fillna(value=0)
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
new_data2['loc_new']=labelencoder.fit_transform(new_data2['localityId'])
new_data2['parking_new']=labelencoder.fit_transform(new_data2['parking'])
new_data2['type_bhk_new']=labelencoder.fit_transform(new_data2['type_bhk'])
x = new_data2[['loc_new', 'bathroom', 'floor', 'maintenanceAmount', 'parking_new', 'property_size', 'totalFloor','type_bhk_new']]
y=new_data2['rent_amount']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

rf = RandomForestRegressor(n_estimators=20)
rf.fit(x_train,y_train)
rf.score(x_test,y_test)*100
def predict_rent(loc_new, bathroom, floor, maintenanceAmount, parking_new, property_size, totalFloor, type_bhk_new):
    # Combine the inputs into the required format
    input_array = [
        loc_new, bathroom, floor, maintenanceAmount, 
        parking_new, property_size, totalFloor, type_bhk_new
    ]
    
    # Predict the rent using the input array
    prediction = rf.predict([input_array])[0]
    
    return prediction



# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Root route to render the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Handle favicon.ico requests
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    loc_new= data['loc_new']
    bathroom = data['bathroom']
    floor = data['floor']
    maintenanceAmount = data['maintenanceAmount']
    parking_new = data['parking_new']
    property_size = data['property_size']
    totalFloor = data['totalFloor']
    type_bhk_new = data['type_bhk_new']
    
    price = predict_rent(loc_new, bathroom, floor, maintenanceAmount,parking_new, property_size,totalFloor, type_bhk_new)
    return jsonify({'price': price})
    #return render_template('result.html') , jsonify({'price': price})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
