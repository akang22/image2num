import os
from flask import Flask, render_template, request, flash, url_for
from werkzeug.utils import redirect, secure_filename
from flask_toastr import Toastr
import time

app = Flask(__name__)
app.secret_key = 'oAQcsFERTqOq6Iua3hvngkCCq33hgzgRp1nwhBkk9agwiZkNOJ'
app.config['UPLOAD_FOLDER'] = os.path.abspath("uploads");
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
toastr = Toastr(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/imageupload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('file uploaded successfully')
            return redirect(url_for('home'))
    flash('Please submit a pdf, png, jpeg, or jpg file.')
    return redirect(url_for('home'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(debug=True)