from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import cv2
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import warnings
import pickle


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True, help="path to the training images")
ap.add_argument("-e", "--testing", required=True,  help="path to the tesitng images")
args = vars(ap.parse_args())

desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

data2 = []
labels2 = []

pridicted_label_set=[]

for imagePath in paths.list_images(args["training"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)
    
model = LinearSVC(C=1000.0, random_state=42, max_iter = 2000)
model.fit(data, labels)

pickle.dump(model,open('model.pkl','wb'))
modelfinal=pickle.load(open('model.pkl','rb'))


for imagePath in paths.list_images(args["testing"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    
    labels2.append(imagePath.split(os.path.sep)[-2])
    data2.append(hist)
    
    prediction = model.predict(hist.reshape(1, -1))
    
    pridicted_label_set.append(prediction[0])


con_matrix=confusion_matrix(labels2,pridicted_label_set,labels=["Live","Fake"])
TP=con_matrix[0][0]
FN=con_matrix[0][1]
FP=con_matrix[1][0]
TN=con_matrix[1][1]

print("True Positive-->The classifier model predicted "+str(TP)+" Live(Positive) samples as Live(Positive)")
print("False Negative-->The classifier model predicted "+str(FN)+" Live(Positive) samples as Fake(Negative)")
print("True Positive-->The classifier model predicted "+str(FP)+" Fake(Negative) samples as Live(Positive)")
print("True Negative-->The classifier model predicted "+str(TN)+" Fake(Negative) samples as Fake(Negative)")
print("Precision of the Linear SVM:", (TP / (TP+FP)))
print("Recall of the Linear SVM:", (TP / (TP+FN)))
print("Accuracy of the Linear SVM:", ((TP + TN) / (TP + TN + FP + FN)))


print("Precision",precision_score(labels2,pridicted_label_set,labels=["Live","Fake"],pos_label="Live"))
print("Recall",recall_score(labels2,pridicted_label_set,labels=["Live","Fake"],pos_label="Live"))
print("Accuracy",accuracy_score(labels2,pridicted_label_set))
