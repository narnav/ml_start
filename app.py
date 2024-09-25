from flask import Flask, jsonify, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

api = Flask(__name__)
file_name="music.csv"
@api.route('/')
def test():
    return render_template('get_predict.html')

@api.route('/predict',methods=['POST'])
def get_prediction():
    # Get input values from the form
    age = int(request.form['age'])
    feature = int(request.form['feature'])
    
    model = joblib.load('our_pridction.joblib')
    prediction = model.predict([[age, feature]])
    
    return jsonify({'prediction': prediction[0]})

@api.route('/send')
def send():
    return render_template('get_predict.html')

@api.route('/ml',methods=['POST'])
def ml():
    # Get input values from the form
    age = int(request.form['age'])
    gender = int(request.form['feature'])
    genre= request.form['genre']

    # Create a DataFrame to hold the new data
    new_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'genre': [genre]
    })
    # Check if the CSV file already exists
    try:
        existing_data = pd.read_csv(file_name)
        # Append the new data to the existing data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        # If the file doesn't exist, just use the new data
        updated_data = new_data

    # Save the updated data to the CSV file
    updated_data.to_csv(file_name, index=False)

    # # gather data
    df = pd.read_csv(file_name)
    # # prepare 2 groups
    X=df.drop(columns=['genre']) # sample features
    Y=df['genre'] # sample output
    X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=.1)
    model = DecisionTreeClassifier()
    model.fit(X_train,Y_train) # load features and sample data
    joblib.dump(model, 'our_pridction.joblib') #binary file
    return 'Hello, World!'

@api.route('/learn')
def learn():
    return render_template('learn.html')

if __name__ == '__main__':
    api.run(debug=True)


