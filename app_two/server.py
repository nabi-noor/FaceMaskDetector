from flask import Flask, render_template, Response, request
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import os


model = load_model('model/model-007.model')
labels_dict={0:'without_mask',1:'with_mask'}


app = Flask(__name__)


def gen_frames():  
    camera = cv2.VideoCapture(0)
    while True:
        
        success, frame = camera.read()  
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        resized=cv2.resize(gray,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)
        print(result)
        if result[0][1] > 0.5:
            cv2.putText(frame, labels_dict[1], (78, 90-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            #os.system("mpg321 thank_you.mp3")
        else:
            
            cv2.putText(frame, labels_dict[0], (78, 90-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            #os.system("mpg321 wear_mask.mp3")
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)