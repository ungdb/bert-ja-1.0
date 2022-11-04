import os
from datetime import datetime
from unittest import result
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, send_from_directory
from bert.predictText import predict
import pandas as pd
from config import *
import docx

from io import TextIOWrapper
import csv

from bert.lib import *
from config import *
from bert.sentimentClassifier import *

import logging

logging.basicConfig(filename='./logs/log.log',
                level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = BertJapaneseTokenizer.from_pretrained(pre_trained_model_name)

model = SentimentClassifier()
model = model.to(device)

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['csv', 'txt', 'docx'])

def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
  
@app.route('/')
def index():
    app.logger.info('--------Index------')
    return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    app.logger.info('--------Begin------')
    if request.method == 'POST':
        try:
            msg = ''
            f = request.files['file']
            print("----------------------------------")
            print(request.files['file'])
            print(request.files.get('file'))
            if f and allowed_file(f.filename):
                # filename = secure_filename(f.filename)
                file_name, file_extension = os.path.splitext(f.filename)
                app.logger.info('File name: ' + file_name)

                if file_extension == '.csv':
                    # csv file
                    result = parseCSV(f)
                elif file_extension == '.txt':
                    # txt file
                    file_name_str = file_name + ".txt"
                    file_path = os.path.join(upload_path, file_name_str)
                    f.save(file_path)

                    result = parseTxt(file_path)
                else:
                    # docx file
                    file_name_str = file_name + ".docx"
                    file_path = os.path.join(upload_path, file_name_str)
                    f.save(file_path)

                    result = parseDocx(file_path)
                
                # Get date time
                date_time = datetime.now()
                date_time_str = date_time.strftime("%Y%m%d%H%M%S")
                file_name_csv = 'ouput_' + date_time_str + '.csv'

                print(date_time_str)
                # Ouput data to csv file
                columns = ['テキスト', 'ラベル']
                outputCsv = pd.DataFrame(result)
                outputCsv.to_csv(os.path.join(upload_path, file_name_csv), encoding='utf8', header=columns, index=False)
                
                app.logger.info('--------End------')
                return render_template("index.html", results=file_name_csv, error=msg)
            else:
                msg = 'Invalid upload only .txt, .csv, .docx.'
        except Exception as e:
            app.logger.error(str(e))
            msg = 'A system error has occurred. Please try again.'
        app.logger.info('--------End------')
        return render_template('index.html', results='', error=msg)
    else:
        app.logger.info('--------End------')
        return redirect('/')

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    app.logger.info('-------------Begin download file-------------')
    path = os.path.join(app.root_path, upload_path)
    app.logger.info('-------------End download file---------------')
    return send_from_directory(directory=path, path=filename, as_attachment=True)

def parseCSV(csv_file):
    app.logger.info('Parse csv file')
    # Use TextIOWrapper to parse the CSV file
    csv_file = TextIOWrapper(csv_file, encoding='utf-8')
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    result = []
    # Loop through the Rows
    for row in csv_reader:
        if len(row) > 0:
            dataRow = []
            dataRow.append(row[0])
            labelVal = predict(device, tokenizer, model, row[0])
            dataRow.append(labelVal)
            result.append(dataRow)

    return result

def parseTxt(path_file):
    result = []
    txt_file = open(path_file, 'r', encoding='utf-8')
    Lines = txt_file.readlines()
    for line in Lines:
        print(line)
        if line and line.strip():
            dataRow = []
            dataRow.append(line.strip())
            labelVal = predict(device, tokenizer, model, line.strip())
            dataRow.append(labelVal)
            result.append(dataRow)

    txt_file.close()

    # Delete the file .txt after finishing processing
    os.remove(path_file)
    return result

def parseDocx(path_file):
    result = []
    doc = docx.Document(path_file)
    all_paras = doc.paragraphs

    for para in all_paras:
        print(para.text)
        if para.text and para.text.strip():
            dataRow = []
            dataRow.append(para.text)
            labelVal = predict(device, tokenizer, model, para.text)
            dataRow.append(labelVal)
            result.append(dataRow)

    # Delete the file .docx after finishing processing
    os.remove(path_file)
    return result

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')