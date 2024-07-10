from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from dataProcessing import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}
COUNTER_FILE = 'counter.txt'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_next_file_number():
    with open(COUNTER_FILE, 'r') as f:
        number = int(f.read().strip())
    with open(COUNTER_FILE, 'w') as f:
        f.write(str(number + 1))
    return number + 1

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/upload", methods=['GET', 'POST'])
def uploadPAGE():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_number = get_next_file_number()
            filename = f"{file_number}{os.path.splitext(file.filename)[1]}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            return redirect(url_for('usersurvey', file_number=file_number))
    return render_template("upload.html")

@app.route("/Processing/<file_number>/<data>")
def processing(file_number, data):
    result = simpleProcessingtest(f"uploads/{file_number}.csv", eval(data))
    return jsonify(result)

@app.route("/survey/<file_number>")
def usersurvey(file_number):
    columns = getdata(f'uploads/{file_number}.csv')
    return render_template("user_page.html", columns=columns, file_number=file_number)

@app.route("/process_survey/<file_number>", methods=["POST"])
def process_survey(file_number):
    survey_data = []
    for key, value in request.form.items():
        survey_data.append(int(value))  # Convert to integer and append to the list

    # Process the survey data and get results
    results = simpleProcessingtest(f"uploads/{file_number}.csv", userX=survey_data)
    print(survey_data)  # Print or process the collected data as needed

    return redirect(url_for('thank_you', results=results))

@app.route("/thank_you")
def thank_you():
    results = request.args.get('results')
    return render_template("thank_you.html", results=results)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
