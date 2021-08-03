from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import os
from gtts import gTTS

model = load_model('model/model-015.model')
labels_dict={0:'with_mask',1:'without_mask'}
myobj = gTTS(text="Please wear your mask", lang="en", slow=False)
myobj.save("wear_mask.mp3")

myobj = gTTS(text="Thank you for wearing mask", lang="en", slow=False)
myobj.save("thank_you.mp3")


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded/image'

@app.route('/')
def upload_f():
	return render_template('index.html')

def finds(img):
    result=model.predict(prepare(img))
    if result[0][0] > 0.5:
        os.system("mpg321 thank_you.mp3")
        return 
    os.system("mpg321 wear_mask.mp3")
    return
def prepare(img):
    img = cv2.imread("uploaded/image/"+img.filename, cv2.IMREAD_GRAYSCALE)
    resized=cv2.resize(img,(100,100))
    normalized=resized/255.0
    reshaped=np.reshape(normalized,(1,100,100,1))
    return reshaped

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
		finds(f)
		return render_template('index.html')

if __name__ == '__main__':
	app.run()
