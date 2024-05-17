import numpy as np
from flask import Flask, request, render_template # type: ignore
import pickle
#Create an app object using the Flask class.
app=Flask(__name__,template_folder='template')
#Load the trained model. (Pickle file)
model1 = pickle.load(open('model.pkl', 'rb'))
#Define the route to be home.
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 
#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')
#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():
    # int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    # features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    features=[[45,0,59,1,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,13.5,2.54,3.6,0.3,19,0.63,140,4,0,1,1,0,0,13.6,1.85,5.2,3,28,1.1,1,136,21.2,4.3,3,0,1,1,1,1,1,0,7.1,5.1,3,1.1,0,1,0,0,105,35,105,0.47]]

    prediction = model1.predict(features)  # features Must be in the form [[a, b]]
    
    # output = round(prediction[0], 2)
    output = prediction[0]
    return render_template('index.html', prediction_text='Output ==> {}'.format(output))
#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).
if __name__ == "__main__":
    app.run(debug = True)