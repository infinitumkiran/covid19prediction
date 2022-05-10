import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from skimage.transform import resize
from werkzeug.utils import secure_filename
import numpy as np
import keras.models
import re
import sys 
import os
import base64
sys.path.append(os.path.abspath("./model"))
from model.load import * 


global graph, model

model, graph = init()


UPLOAD_FOLDER = 'static/images/'

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classes=["COVID","healthy","Others"]

def model_predict(img_path,model):
    img=load_img(img_path,target_size=(224,224))
    img=img_to_array(img)
    img=resize(img,(100,100))
    img=np.reshape(img,(1,)+(100,100,3))
    return classes[np.argmax(model.predict(img))]

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predict():
    f=request.files['file']
    basepath=os.path.dirname(__file__)
    file_path=os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
    f.save(file_path)
    preds=model_predict(file_path,model)
    return preds

if __name__ == '__main__':
    app.run(debug=True, port=5005)