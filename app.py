# Import the libraries
import numpy as np
from flask import Flask, request, render_template
import joblib

# Create the Flask app and load the trained model
app = Flask(__name__)
model = joblib.load('model_dtgs.pkl')

# Define the '/' root route to display the content from index.html
@app.route('/')
def home():
    return render_template('index.html')

    # Define the '/predict' route to:
# - Get form data and pre-process the data with functions
# - Convert form data to numpy array
# - Pass form data to model for prediction

# define function to pre-process data from drop-down menu --> array with shape (1,3)
def preprocess_onehot(number):
    options = 3
    number = int(number)
    lst = [0 for i in range(options)]
    lst[number] = 1
    return lst 

# define function to pre-process data from age --> array with shape (1,6) representing age bracket
def preprocess_age(age):
    """
    1-5       81 [1,0,0,0,0,0]
    6-10      51 [0,1,0,0,0,0]
    11-15    353 [0,0,1,0,0,0]
    16-20    278 [0,0,0,1,0,0]
    21-25    374 [0,0,0,0,1,0]
    26-30     68 [0,0,0,0,0,1]
    """
    age = int(age)
    if age >= 26:
        return [0,0,0,0,0,1]
    if age < 26 and age >= 21:
        return [0,0,0,0,1,0]
    if age <21 and age >= 16:
        return [0,0,0,1,0,0]
    if age < 16 and age >= 11:
        return [0,0,1,0,0,0]
    if age < 11 and age >= 6:
        return [0,1,0,0,0,0]
    return [1,0,0,0,0,0]

@app.route('/predict',methods=['POST'])
def predict():

    form_data = request.form.to_dict()
    """ 
    print(request.form) gives
    ImmutableMultiDict([('gender', '0'), ('instiution_type', '0'), ('it_student', '0'), ('location', '0'), ('load_shedding', '0'), ('internet_type', '0'), ('lms', '0'), 
    ('education_level', '0'), ('financial_condition', '0'), ('class_duraion', '0'), ('device', '0'), ('network', '0'), ('age', '25')])
    """
    features = []
    preprocess_items = ['education_level','financial_condition', 'class_duraion', 'device', 'network']
    for k, v in form_data.items(): 
        if k not in preprocess_items and k != 'age':
            features += [v]
        if k in preprocess_items:
            lst = preprocess_onehot(v)
            features += lst 
        if k == 'age':
            age_lst = preprocess_age(v)
            features += age_lst
    features = [np.array(features)]
    #print(form_data)
    #print(features)
    prediction = model.predict(features)
    #print(prediction)

	# Format prediction text for display in "index.html"
    adap_lvl = ['low', 'moderate', 'high']
    return render_template('index.html', adaptivity_level="Student's adaptivity level to online learning is {}.".format(adap_lvl[prediction[0]]))

if __name__ == '__main__':
    app.run(debug=False)