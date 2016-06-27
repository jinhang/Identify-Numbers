# -*- coding:utf-8 -*-

from flask import Flask,request
from scipy import misc
from sklearn.externals import joblib

app = Flask(__name__)



@app.route('/upload', methods=['POST'])
def upload():

	f = request.files['file']
	im = misc.imread(f)
		
	img = im.reshape((1,784))

	clf = joblib.load('model/ok.m')

	l = clf.predict(img)

	return 'predict: %s ' % (l[0])

@app.route('/')
def index():
	return '''
    	<!doctype html>
    	<html>
    	<body>
    	<form action='/upload' method='post' enctype='multipart/form-data'>
      		<input type='file' name='file'>
        	<input type='submit' value='Upload'>
    	</form>
    	'''


if __name__ == '__main__':
	app.run()