import os
from datetime import datetime
from bert.classification import sentence_classification

from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
import pandas as pd
from config import *
import csv
import docx
from io import TextIOWrapper

from bert.lib import *
from config import *

import logging

logging.basicConfig(filename='./logs/log.log',
                level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

"""------------Start Use classification.py file--------------------"""
tokenizer = BertJapaneseTokenizer.from_pretrained(pre_trained_model_name, truncation=True)
model = BertForSequenceClassification.from_pretrained(pre_trained_model_name, return_dict=True).to(device)
nlp = ja_core_news_sm.load()
"""------------End Use classification.py file--------------------"""

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
            if f and allowed_file(f.filename):
                # filename = secure_filename(f.filename)
                file_name, file_extension = os.path.splitext(f.filename)
                app.logger.info('File name: ' + file_name)

                chartsOutput = [{"key": "Label_0","value": 0 }, {"key": "Label_1","value": 0 }]

                if file_extension == '.csv':
                    # csv file
                    result, dataCharts = parseCSV(f)

                    rcLength = len(result)
                    # Cal % of records as 0
                    chartsOutput[0]['value'] = dataCharts['lb0']/rcLength*100
                    rcLength = len(result)

                    # Cal % of records as 1
                    chartsOutput[1]['value'] = dataCharts['lb1']/rcLength*100

                elif file_extension == '.txt':
                    # txt file
                    file_name_str = file_name + ".txt"
                    file_path = os.path.join(upload_path, file_name_str)
                    f.save(file_path)

                    result, dataCharts = parseTxt(file_path)

                    rcLength = len(result)
                    # Cal % of records as 0
                    chartsOutput[0]['value'] = dataCharts['lb0']/rcLength*100
                    rcLength = len(result)

                    # Cal % of records as 1
                    chartsOutput[1]['value'] = dataCharts['lb1']/rcLength*100
                else:
                    # docx file
                    file_name_str = file_name + ".docx"
                    file_path = os.path.join(upload_path, file_name_str)
                    f.save(file_path)

                    result, dataCharts = parseDocx(file_path)

                    rcLength = len(result)
                    # Cal % of records as 0
                    chartsOutput[0]['value'] = dataCharts['lb0']/rcLength*100
                    rcLength = len(result)

                    # Cal % of records as 1
                    chartsOutput[1]['value'] = dataCharts['lb1']/rcLength*100
                
                # Get date time
                date_time = datetime.now()
                date_time_str = date_time.strftime("%Y%m%d%H%M%S")
                file_name_csv = 'ouput_' + date_time_str + '.csv'

                # Ouput data to csv file
                outputCsv = pd.DataFrame(result)
                outputCsv.to_csv(os.path.join(upload_path, file_name_csv), encoding='utf8', header=header_name, index=False)
                
                app.logger.info('--------End------')
                return jsonify(results=chartsOutput, filename=file_name_csv, error=msg)
                # return render_template("index.html", results=file_name_csv, error=msg)
            else:
                msg = 'Invalid upload only .txt, .csv, .docx.'
        except Exception as e:
            app.logger.error(str(e))
            msg = 'A system error has occurred. Please try again.'
        app.logger.info('--------End------')
        return jsonify(results=chartsOutput, filename='', error=msg)
        # return render_template('index.html', results='', error=msg)
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

    # data ouput chart
    dataCharts = {'lb0': 0, 'lb1': 0 }

    # Loop through the Rows
    for row in csv_reader:
        if len(row) > 0:
            dataRow = []
            dataRow.append(row[0])
            # labelVal = predict(device, tokenizer, model, row[0])
            predicted_class_id, labels, logits_max = sentence_classification(device, tokenizer, model, nlp, row[0])
            dataRow.append(predicted_class_id)
            result.append(dataRow)
            if predicted_class_id == 0:
                dataCharts['lb0'] += 1
            else:
                dataCharts['lb1'] += 1

    return result, dataCharts

def parseTxt(path_file):
    result = []
    # data ouput chart
    dataCharts = {'lb0': 0, 'lb1': 0 }
    txt_file = open(path_file, 'r', encoding='utf-8')
    Lines = txt_file.readlines()
    for line in Lines:
        # print(line)
        if line and line.strip():
            dataRow = []
            dataRow.append(line.strip())
            # labelVal = predict(device, tokenizer, model, line.strip())
            predicted_class_id, labels, logits_max = sentence_classification(device, tokenizer, model, nlp, line.strip())
            print(predicted_class_id, labels, logits_max)
            dataRow.append(predicted_class_id)
            result.append(dataRow)
            if predicted_class_id == 0:
                dataCharts['lb0'] += 1
            else:
                dataCharts['lb1'] += 1

    txt_file.close()

    # Delete the file .txt after finishing processing
    os.remove(path_file)
    return result, dataCharts

def parseDocx(path_file):
    result = []
    # data ouput chart
    dataCharts = {'lb0': 0, 'lb1': 0 }
    doc = docx.Document(path_file)
    all_paras = doc.paragraphs

    for para in all_paras:
        # print(para.text)
        if para.text and para.text.strip():
            dataRow = []
            dataRow.append(para.text)
            # labelVal = predict(device, tokenizer, model, para.text)
            predicted_class_id, labels, logits_max = sentence_classification(device, tokenizer, model, nlp, para.text)
            print(predicted_class_id, labels, logits_max)
            dataRow.append(predicted_class_id)
            result.append(dataRow)
            if predicted_class_id == 0:
                dataCharts['lb0'] += 1
            else:
                dataCharts['lb1'] += 1

    # Delete the file .docx after finishing processing
    os.remove(path_file)
    return result, dataCharts

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')