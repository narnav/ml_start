# end point that predict user gener

@api.route('/predict',methods=['POST'])
def get_prediction():
    # Get input values from the form
    age = int(request.form['age'])
    feature = int(request.form['feature'])
    
    model = joblib.load('our_pridction.joblib')
    prediction = model.predict([[age, feature]])
    
    return jsonify({'prediction': prediction[0]})

## update CSV and train the model
@api.route('/ml',methods=['POST'])
def ml():
    # Get input values from the form
    age = int(request.form['age'])
    gender = int(request.form['feature'])
    genre= request.form['genre']
