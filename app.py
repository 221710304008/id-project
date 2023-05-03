from flask import Flask, render_template, request, redirect, send_from_directory
import os
import json

from model import get_amount
# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def get_output():


# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')

# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = file.filename
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        try:
            check_amount = get_amount(os.path.join(app.config['UPLOAD_FOLDER']))
            return render_template("result.html", filename=filename, is_check=True, check_amount=check_amount)
        except:
            check_amount = get_amount(os.path.join(app.config['UPLOAD_FOLDER']))
            return render_template("result.html", filename=filename, is_check=False, check_amount=0)

@app.route('/upload', methods=['GET'])
def result_page():
    return render_template('result.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
    


if __name__ == 'main':
    app.run(
        host="0.0.0.0",
        port=int("8080"),
        debug=True
    )