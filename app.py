from flask import Flask 
from flask import render_template
from flask import request
import pickle
import cv2
from PIL import Image
import io
import re
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import cv2
import json
import base64

app = Flask(__name__,template_folder='templates') 

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def main():
	return(render_template('main.html',pred=''))

@app.route('/predict', methods=['POST','GET'])
def predict():
    for imagePath in paths.list_images('testimage'):
            desc = LocalBinaryPatterns(24, 8)
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #gray=cv2.Canny(gray,30,230)
            hist = desc.describe(gray)
            prediction = model.predict(hist.reshape(1, -1))
            return render_template('main.html',pred='The fingerprint is '+ prediction[0])
            
            



@app.route('/convert', methods=['POST','GET'])
def convert():
    output = request.get_json()
    result = json.loads(output)
    print(result.keys())
    with open("trimmed.txt","w") as output:
        output.write(result["username"][22:])

    file = open('trimmed.txt', 'rb')
    byte = file.read()
    file.close()
    
    decodeit = open('D:/6th Sem/Minor proj/Project(2)/testimage/fingerprintimg.jpg', 'wb')
    decodeit.write(base64.b64decode((byte)))
    decodeit.close()


if __name__=="__main__":
 app.run(debug=True)

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">