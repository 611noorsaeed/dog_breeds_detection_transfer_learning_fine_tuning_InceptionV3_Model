from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_sqlalchemy import SQLAlchemy


# Create an app
app = Flask(__name__)
app.secret_key = "Secret Key"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/signdb"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)


# Route for signup page
@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        new_signup = Signup(username=username, email=email)
        db.session.add(new_signup)
        db.session.commit()
        return render_template('index.html', signup_message='User signed up successfully!')


class_labels = ['beagle', 'bulldog', 'dalmatian', 'german-shepherd', 'husky', 'labrador-retriever', 'poodle',
                'rottweiler']


# Function to preprocess and predict
def predict_and_display(img):
    # Load the model
    model = tf.keras.models.load_model('models/fine_tuned_inception.h5')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence_level = np.max(predictions)
    predicted_class_name = class_labels[predicted_class]

    return predicted_class_name, confidence_level


# Route to the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)
            img = image.load_img(filepath, target_size=(224, 224))


            predicted_class, confidence = predict_and_display(img)



            return render_template('index.html', image_path=filepath, actual_label=predicted_class,
                                   predicted_label=predicted_class, confidence=confidence)
    return render_template('index.html', message='Upload an image')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


if __name__ == '__main__':
    app.run(debug=True)
