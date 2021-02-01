from flask import Flask,render_template
from flask import request
import numpy as np
import pickle

import os
from flask import send_from_directory
app=Flask(__name__,template_folder='template')

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/cancer")
def cancer():
    return render_template("cancer.html")
    
@app.route("/diabetes")
def diabetes():
    #if form.validate_on_submit():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/liver")
def liver():
    #if form.validate_on_submit():
    return render_template("liver.html")

@app.route("/kidney")
def kidney():
    #if form.validate_on_submit():
    return render_template("kidney.html")
    
def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8):#Diabetes
        loaded_model = pickle.load(open('model_diabetes.pkl','rb'))
        result = loaded_model.predict(to_predict)
    elif(size==30):#Cancer
        loaded_model = pickle.load(open('model_cancer.pkl','rb'))
        result = loaded_model.predict(to_predict)
    elif(size==24):#Kidney
        loaded_model = pickle.load(open('model_kidney.pkl','rb'))
        result = loaded_model.predict(to_predict)
    elif(size==10):#liver
        loaded_model = pickle.load(open('model_liver.pkl','rb'))
        result = loaded_model.predict(to_predict)
    elif(size==11):#Heart
        loaded_model = pickle.load(open('model_heart.pkl','rb'))
        result =loaded_model.predict(to_predict)
    return result[0]

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list)==30):#Cancer
            result = ValuePredictor(to_predict_list,30)
        elif(len(to_predict_list)==8):#Daibetes
            result = ValuePredictor(to_predict_list,8)
        elif(len(to_predict_list)==24):#Kidney
            result = ValuePredictor(to_predict_list,24)
        elif(len(to_predict_list)==11):#Heart
            result = ValuePredictor(to_predict_list,11)
            #if int(result)==1:
            #   prediction ='diabetes'
            #else:
            #   prediction='Healthy' 
        elif(len(to_predict_list)==10):#liver
            result = ValuePredictor(to_predict_list,10)
    if(int(result)==1):
        prediction='Sorry ! Suffering'
    else:
        prediction='Congrats ! you are Healthy' 
    return(render_template("result.html", prediction=prediction))


if __name__ == "__main__":
    app.run(debug=True)
