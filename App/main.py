import os
import importlib
from flask import Flask, render_template, request, flash, url_for
from werkzeug.utils import redirect, secure_filename
from flask_toastr import Toastr
from PIL import Image
from flask import Flask, request, jsonify
from nn_utils import transform_image, get_prediction

app = Flask(__name__)
app.secret_key = 'oAQcsFERTqOq6Iua3hvngkCCq33hgzgRp1nwhBkk9agwiZkNOJ'
app.config['UPLOAD_FOLDER'] = os.path.abspath("uploads");
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
toastr = Toastr(app)

@app.route("/")
def home():
    return render_template("index.html", number_guess=None)

@app.route('/imageupload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:  
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            print(prediction)
            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
            return render_template("index2.html", number_guess=data)
        except:
            return jsonify({'error': 'error during prediction'})
    flash('Please submit a png, jpeg, or jpg file.')
    return redirect(url_for('home'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(debug=True)