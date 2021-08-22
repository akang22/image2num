import os
import importlib
from flask import Flask, render_template, request, flash, url_for
from werkzeug.utils import redirect, secure_filename
from flask_toastr import Toastr
from PIL import Image

app = Flask(__name__)
app.secret_key = 'oAQcsFERTqOq6Iua3hvngkCCq33hgzgRp1nwhBkk9agwiZkNOJ'
app.config['UPLOAD_FOLDER'] = os.path.abspath("uploads");
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
toastr = Toastr(app)

@app.route("/")
def home():
    return render_template("index.html", number_guess=None)

@app.route('/imageupload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash('file uploaded successfully')
            if filename.rsplit('.', 1)[1].lower() != 'jpg':
                my_image = Image.open(filepath)
                my_image = my_image.convert('RGB')
                oldfilepath = filepath
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename.rsplit('.', 1)[0] + '.jpg')
                my_image.save(filepath)
                os.remove(oldfilepath)
            model = importlib.import_module('model')
            string_output = model.get_translation(filepath)
            return render_template("index.html", number_guess=string_output)
    flash('Please submit a png, jpeg, or jpg file.')
    return redirect(url_for('home'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(debug=True)
