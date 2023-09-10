from flask import Flask, render_template, request
from flask import Flask, render_template, request, send_from_directory
from sklearn.model_selection import train_test_split
from PIL import Image
from twilio.rest import Client
import numpy as np
import tensorflow as tf
import cv2,os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
print(os.getcwd())

app = Flask(__name__)
img_size = 256
model = tf.keras.models.load_model('C:/Users/HP/Desktop/Intelligent-Approach-for-Classification-of-Osteoporosis-master/Flask App/best_model.h5')
categories = ['Normal', 'Doubtful', 'Moderate', 'Mild', 'Severe']


def auth(prediction, precaution, phone):
    account_sid = 'AC986aedccd4ff17e11b4cfcd5d91c6493'
    auth_token = '4f5adf501fb32f458912b99f3dc3a3c5'
    number = '+13613155527'
    phone = "+91" + phone
    print(phone)
    dest = phone
    precautions = ""
    for i in precaution:
        precautions += i + "\n"
    msg_body = "Predicted Category:"+prediction + "\n" +"Precautons:\n"+ precautions
    
    client = Client(account_sid, auth_token)
    msg = client.messages.create(
        body = msg_body,
        from_= number,
        to = dest
    )
    
def predict(img_path):
    img = Image.open(img_path).convert('L')  # convert to grayscale
    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.0  # normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    # Prediction 
    predictions_test_single = model.predict(img)
    predicted_category = categories[np.argmax(predictions_test_single)]
    return predicted_category

def get_precaution(prediction):
    precautions = {
        'Normal': ["Maintain a balanced and nutritious diet to support bone health.",
                   " Engage in regular weight-bearing exercises like walking, jogging, or dancing to strengthen bones.",
                   " Avoid excessive alcohol consumption and smoking, as they can contribute to bone loss.", 
                   " Get regular check-ups and bone density tests as recommended by your healthcare provider."],
        'Doubtful': [" Eat food which contains Vitamin D."," Follow any additional tests or screenings recommended by your healthcare provider."," Maintain a healthy lifestyle with a focus on nutrition, exercise, and overall well-being."],
        'Moderate': [" Take necessary precautions to prevent falls, such as removing hazards at home, using assistive devices, and ensuring proper lighting."," Follow the recommendations of your healthcare provider regarding medication, supplements, and physical therapy."," Engage in exercises that focus on balance, strength, and flexibility to reduce the risk of fractures."," Consider modifications in daily activities to prevent excessive strain on the bones."],
        'Mild': [" Take necessary precautions similar to those for the moderate category."," Follow the advice of your healthcare provider regarding lifestyle modifications, medication, and therapeutic interventions."," Engage in exercises that are appropriate for your condition and focus on improving bone strength and flexibility."," Ensure an adequate intake of calcium, vitamin D, and other essential nutrients for bone health."],
        'Severe': [" Seek immediate medical attention and follow the guidance of healthcare professionals.",
                   " Adhere strictly to the prescribed treatment plan, including medication, therapy, and lifestyle modifications.",
                   " Take precautions to minimize the risk of falls and fractures, such as using assistive devices and making necessary home modifications.",
                   " Engage in physical activities as recommended by your healthcare provider, considering the limitations of your condition."]
    }
    return precautions.get(prediction)

@app.route("/", methods=['GET', 'POST'])
def index():
    global prediction
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file found"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        if file:
            file_path = "C:/Users/HP/Desktop/Intelligent-Approach-for-Classification-of-Osteoporosis-master/Flask App/static/uploads/" + file.filename
            phone = request.form['phone']
            file.save(file_path)
            prediction = predict(file_path)
            precaution = get_precaution(prediction)
            print(phone)
            auth(prediction, precaution, phone)
            # res = prediction + " detected " + precaution
            return render_template("result.html", prediction = prediction, precaution=precaution, image_file=file.filename)
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)

@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0',port='5000', debug=True)

