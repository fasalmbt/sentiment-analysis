import os
import pickle
from flask import Flask, jsonify, request, redirect, render_template

app = Flask(__name__)

vectorizer = pickle.load(open('./models/vectorizer.sav', 'rb'))
classifier = pickle.load(open('./models/classifier.sav', 'rb'))

@app.route('/')
def main():
	return render_template('index.html')

@app.route('/analysis', methods=['POST', 'GET'])
def sentiment():
	if request.method == 'GET':
		word = request.args.get('word')
		if word:
			word_vector = vectorizer.transform([word])
			result = classifier.predict(word_vector)
			return jsonify({'Sentiment': result[0], 'Word': word, 'Status code': 200})
		else:
			return "<center> Please provide a word.</center>"
		return jsonify({'error':'internal server error', 'Status code': 500})


if __name__ == '__main__':
	app.run()
