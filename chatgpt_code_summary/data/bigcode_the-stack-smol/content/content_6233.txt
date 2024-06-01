from flask import Flask
from flask import request, session, render_template, json, Response, jsonify, make_response, send_file, redirect, url_for
import requests
import xml.etree.ElementTree as ET
import lxml
import pandas as pd
import re

app = Flask(__name__)

@app.route('/')

def index():
	return render_template('process_fulltext.html')


@app.route('/process_fulltext', methods = ['GET', 'POST'])

def process_fulltext():
	upload = request.files.get('file', '').read() #puts the uploaded file (in the request)
	url = 'http://localhost:8070/api/processFulltextDocument'
	files = dict(input=upload, teiCoordinates="biblStruct")
	r = requests.post(url, files=files)
	return render_template('process_fulltext.html', r=r.text)

# takes a string and removes xml element inside
def clean(text):
	text = re.sub("<[^>]+>","", text)
	text = re.sub("^\s+|\s+$","", text)
	return text


#parses the tei document and creates list of dictionaries out of tei elements
def parse_tei(xml):
	#data = open(xml)
	data = xml.split('\n')
	refs = []
	ref = []
	start = False
	title = ""
	name = ""
	date = ""
	names = []
	year = ""
	#art_name = re.sub(".*\/","")
	old_ref = {"title": "", "name": "", "date": "", "year_pub": ""}
	for line in data:
		if re.match(".*<date",line) and start == False:
			
			year = re.sub(".*when\=\"","",line)
			year = re.sub("\".*","",year)[0:4]

		if start == False and re.match(".*<back",line):
			start = True
		if start == False:
			continue
			
		if re.match(".*<biblStruct",line):
			
			if title == "":
				continue
			ref = {"title": title, "name": names, "date": date, "year_pub": year}   
			
			if ref["title"] == old_ref["title"]:
			
			
				continue
			else:
				refs.append(ref)
				olf_ref = ref
				names = []
		if re.match(".*<title.*type=\"main\"",line):
			title = clean(line)
		if re.match(".*<persName",line):
			forename = re.sub("<\/forename.*","",line)
			forename = clean(forename)
			
			surname = re.sub(".*<surname","",line)
			surname = clean(surname)
			surname = re.sub(">",". ",surname)
			name = forename+surname
			names.append(name)
			
		if re.match(".*<date",line):
			date = re.sub(".*when\=\"","",line)
			date = re.sub("\".*","",date)
			date = date[0:4]
	
	return refs


# sends request to grobid api to process the pdf and returns data in dataframe to template view
@app.route('/process_references', methods = ['GET', 'POST'])

def process_references():
	upload = request.files.get('file', '').read() #puts the uploaded file (in the request)
	url = 'http://localhost:8070/api/processFulltextDocument'
	files = dict(input=upload, teiCoordinates="biblStruct")
	r = requests.post(url, files=files)
	tei_list = parse_tei(r.text)
	# increase the column width of pd (standard is only 50px)
	pd.set_option('display.max_colwidth', -1)
	df1 = pd.DataFrame(tei_list)
	# removing year_pub column
	df1 = df1.drop('year_pub', axis=1)
	df2 = df1.to_json()
	df1 = df1.to_html()
	# changing css class in html for dataframe output
	df1 = re.sub("dataframe", "myTable", df1)
	return render_template('process_fulltext.html', df1=df1, df2=df2)
	

if __name__ == '__main__':
	app.run(debug=True)