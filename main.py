from flask import Flask, render_template, request
import os
import pickle
import numpy as np
import tensorflow as tf


image_folder = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = image_folder


scaler = pickle.load(open('scaler_ran', 'rb'))

filename = os.path.join(app.config['UPLOAD_FOLDER'], 'levitated_cylinders.png')

plus_arrow = os.path.join(app.config['UPLOAD_FOLDER'], 'Plus_arrow.png')

minus_arrow = os.path.join(app.config['UPLOAD_FOLDER'], 'Minus_arrow.png')


@app.route('/')
def enter_parameters():
	return render_template('mag_in.html', Fig = 'Fig.1 - Parameters of levitated cylinders', figure_geo = filename, arrow_J_1 = minus_arrow, arrow_J = plus_arrow, 
	arrow_F = plus_arrow, J = 1, J_1 = -1, predicted_force = 11.546, R = 10, h = 10, R_1 = 10, h_1 = 10, xi = 10)
@app.route('/predict', methods = ['POST'])
def result():
	model = tf.keras.models.load_model('trained_model_with_random')
	features = [np.float32(para) for para in request.form.values()]
	model_features = np.array([[features[4], features[5], features[1], features[2], features[6]]])
	model_features = scaler.transform(model_features)
	mag_force = round(-1*features[0]*features[3]*model.predict(model_features)[0][0],3)

	return render_template('mag_in.html', Fig = 'Fig.2 - Schematic of predicted results', predicted_force = mag_force, figure_geo = filename, 
	arrow_J_1 = minus_arrow if features[3] < 0 else plus_arrow, arrow_J = minus_arrow if features[0] < 0 else plus_arrow,
	arrow_F = minus_arrow if mag_force < 0 else plus_arrow, J = features[0], R = features[1], h = features[2],
	 J_1 = features[3], R_1 = features[4], h_1 = features[5], xi = features[6])


if __name__ == '__main__':
	app.run(debug=True)