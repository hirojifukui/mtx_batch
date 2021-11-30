from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
import tensorflow as tf

import secrets
import os
import stat
from PIL import Image
from flask import render_template, request, redirect, jsonify, make_response, flash, url_for, Blueprint
from app import app, db, bcrypt, mail
from app.forms import (LoginForm, RegistrationForm, UpdateAccountForm, 
						RequestResetForm, ResetPasswordForm, UploadForm, TestDateForm)
from app.models import User
from datetime import datetime
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
from werkzeug.utils import secure_filename
from pyzbar.pyzbar import decode
import cv2 as cv
import numpy as np
import pymongo
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
from keras.models import model_from_json
from keras.preprocessing import image
import json
from keras.models import load_model

np.random.seed(42)
tf.random.set_seed(42)

# @login_manager.load_user
# def load_user(user):
#     return User.get(user)

class CTCLayer(keras.layers.Layer):
	def __init__(self, name=None, **kwargs):
		#self.name = name
		super(CTCLayer, self).__init__(name=name, **kwargs)
		self.loss_fn = keras.backend.ctc_batch_cost

	def call(self, y_true, y_pred):
		batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
		input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
		label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

		input_length = input_length * \
			tf.ones(shape=(batch_len, 1), dtype="int64")
		label_length = label_length * \
			tf.ones(shape=(batch_len, 1), dtype="int64")
		loss = self.loss_fn(y_true, y_pred, input_length, label_length)
		self.add_loss(loss)
		# At test time, just return the computed predictions.
		return y_pred

	def get_config(self):
		config = super(CTCLayer, self).get_config()
		config.update({"name": self.name})
		return config


#parent_dir = "F:/Google Drive/Pythonapp/"
date = "2020-01-01"
#Restore trained network weight
#model = model_from_json(open('F:/Google Drive/User_Backup/model/cnn_model4.json').read())
#model.load_weights('F:/Google Drive/User_Backup/model/cnn_model_weights4.hdf5')
#model = load_model("F:/Google Drive/User_Backup/model/trained_20201016.h5")
#model_t = load_model("cnn/models/tens_digit_20201013.h5")
#model_oo = load_model("cnn/models/trained_20201014.h5")
#model_o = load_model("cnn/models/ones_digit_20201013.h5")
#model = load_model("cnn/lfs/25x44_2digit_deep_2020604.h5")
#model_single = load_model("cnn/lfs/25x44_1digit_20201018.h5")
#model_s = load_model("cnn/models/single_digit_25x25_32_64_20210114.h5")

path = ""; secure_files = []; form_id =""

model_path = os.path.join(app.root_path, 'static',
						  'model', "CNN+RNN_20211122E17R1C13.h5")
model = tf.keras.models.load_model(
	model_path, custom_objects={'CTCLayer': CTCLayer})

prediction_model = keras.models.Model(
	model.get_layer(name="image").input, model.get_layer(name="dense2").output)

@app.route("/")
def index():
	return render_template("public/index.html")

@app.route("/home")
def home():
	return render_template("public/index.html")

@app.route("/about")
def about():
	return render_template("public/about.html")

@app.route("/login", methods =["GET", "POST"])
def login():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = LoginForm()
	if form.validate_on_submit():
		user = User.query.filter_by(email = form.email.data).first()
		if user and bcrypt.check_password_hash(user.password, form.password.data):
			login_user(user, remember=form.remember.data)
			next_page = request.args.get('next')
			#if not is_safe_url(next_page):
			#     return flask.abort(400)
			return redirect(next_page) if next_page else redirect(url_for('home'))
		else:
			flash('Login Unsuccessful. Please check email and password', 'danger')
	# return render_template(url_for('login'), title="Login", form = form)
	return render_template("public/login.html", title="Login", form = form)

def save_picture(form_picture, email_address):
	random_hex = secrets.token_hex(8)
	_, f_ext = os.path.splitext(form_picture.filename)
	#picture_fn = random_hex + f_ext
	if len(email_address) < 12:
		ln = len(email_address)
	else:
		ln = 12
	picture_fn = email_address[:ln]+random_hex[:9] + f_ext
	picture_path = os.path.join(app.root_path, 'static/img/thumnail', picture_fn)
	output_size = (125, 125)
	i = Image.open(form_picture)
	i.thumbnail(output_size)
	i.save(picture_path)
	return picture_fn

@app.route("/register", methods =["GET", "POST"])
def register():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = RegistrationForm()
	if form.validate_on_submit():
		hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
		user = User(username = form.username.data, email = form.email.data, password = hashed_password)
		db.session.add(user)
		db.session.commit()

		path = os.path.join(app.root.path, 'static', 'img', 'upload', form.username.data)
		try:
			os.makedirs(path, exist_ok = True)
			print("Directry '%s' created successfully" %form.username.data)
		except OSError as error:
			print("Directory 's%' can not be created")
		flash('Your account has been created. You are able to login.', 'success')
		return redirect(url_for("login"))
	return render_template('public/registration.html', title="Register", form = form)

@app.route("/logout")
def logout():
	logout_user()
	return redirect(url_for('home'))

def delete_picture(current_picture):
	picture_path = os.path.join(app.root_path, 'static', 'img', 'thumnail', current_picture)
	print("Deleting: ", picture_path)
	os.chmod(picture_path, stat.S_IWRITE)
	r = os.remove(picture_path)
	return

@app.route("/account", methods =["GET", "POST"])
@login_required
def account():
	form = UpdateAccountForm()
	if form.validate_on_submit():
		if form.picture.data:
			picture_file = save_picture(form.picture.data, form.email.data)
			print("Current image: ", current_user.image_file, "New image: ", picture_file)
			if current_user.image_file != "default.jpg":
				delete_picture(current_user.image_file)
			current_user.image_file = picture_file
		current_user.username = form.username.data
		current_user.email = form.email.data
		db.session.commit()
		flash('Your account has been updated.', 'success')
		return redirect(url_for('account'))
	elif request.method == 'GET':
		form.username.data = current_user.username
		form.email.data = current_user.email
	image_file = url_for('static', filename='img/thumnail/'+current_user.image_file)
	return render_template("public/account.html", title="Account", image_file=image_file, form = form)

@app.route("/profile/<username>")
def profile(username):
	users = {
	"mitsuhiko": {
		"name": "Armin Ronacher",
		"bio": "Creatof of the Flask framework",
		"twitter_handle": "@mitsuhiko"
	},
	"gvanrossum": {
		"name": "Guido Van Rossum",
		"bio": "Creator of the Python programming language",
		"twitter_handle": "@gvanrossum"
	},
	"elonmusk": {
		"name": "Elon Musk",
		"bio": "technology entrepreneur, investor, and engineer",
		"twitter_handle": "@elonmusk"
	}
	}	
	user = None
	if username in users:
		user = users[username]
	return render_template("public/profile.html", username = username, user = user)

## @app.route("/json", methods=["POST"])
## def json_example():
##	if request.is_json:
##		req = request.get_json()
##		response_body = {
##			"message": "JSON received!",
##			"sender": req.get("name")
##		}
##
##		res = make_response(jsonify(response_body), 200)
##		return res
##	else:
##		return make_response(jsonify({"message": "Request body must be JSON"}), 400)

## @app.route("/guestbook")
## def guestbook():
## return render_template("public/guestbook.html")

## @app.route("/guestbook/create-entry", methods=["POST"])
##def create_entry():
##
##    req = request.get_json()
##
##    print(req)
##
##    res = make_response(jsonify(req), 200)
##
##    return res

def send_reset_email(user):
	token = user.get_reset_token()
	msg = Message('Password Reset Request',
				  sender='support@us.myrisu.com',
				  recipients=[user.email])
	msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}
If you did not make this request then simply ignore this email and no changes will be made.
'''
	mail.send(msg)

@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = RequestResetForm()
	if form.validate_on_submit():
		user = User.query.filter_by(email=form.email.data).first()
		send_reset_email(user)
		flash('An email has been sent with instructions to reset your password.', 'info')
		return redirect(url_for('login'))
	return render_template('public/reset_request.html', title='Reset Password', form=form)


@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	user = User.verify_reset_token(token)
	if user is None:
		flash('That is an invalid or expired token', 'warning')
		return redirect(url_for('reset_request'))
	form = ResetPasswordForm()
	if form.validate_on_submit():
		hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
		user.password = hashed_password
		db.session.commit()
		flash('Your password has been updated! You are now able to log in', 'success')
		return redirect(url_for('login'))
	return render_template('public/reset_token.html', title='Reset Password', form=form)
	
def allowed_file(filename):
	ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/testdate", methods=['GET', 'POST'])
@login_required
def testdate():
	global date, path
	if current_user.is_authenticated == False:
		flash('Login required. Please login before enter the test date', 'danger')
		return redirect(url_for('login'))
	form = TestDateForm()
	if form.validate_on_submit():
		#print ("Files: ", form.testdate.data)
		#print("Currnet_user name", current_user.username)
		date = str(form.testdate.data)
		msg = "Test Date is " + date
		print(msg)
		flash(msg)
		#path = os.path.join(parent_dir, "uploads", current_user.username, date)
		path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date)
		#print("Path:", path)
		#print(os.path.exists(path))
		if os.path.exists(path) == False:
			try:
				os.makedirs(path, exist_ok = True)
				print("Directry '%s' created successfully" %path)
			except OSError as error:
				print("Directory 's%' can not be created")    
		return redirect(url_for('upload_files'))
	elif request.method == 'GET':
		return render_template("public/testdate.html", title="Test Date", form = form)
	else:
		flash('Else')
		return render_template("public/testdate.html", title="Test Date", form = form)


#Resize image without distortion
def distortion_free_resize(image, img_size):
	w, h = img_size
	image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

	# Check tha amount of padding needed to be done.
	pad_height = h - tf.shape(image)[0]
	pad_width = w - tf.shape(image)[1]

	# Only necessary if you want to do same amount of padding on both sides.
	if pad_height % 2 != 0:
		height = pad_height // 2
		pad_height_top = height + 1
		pad_height_bottom = height
	else:
		pad_height_top = pad_height_bottom = pad_height // 2

	if pad_width % 2 != 0:
		width = pad_width // 2
		pad_width_left = width + 1
		pad_width_right = width
	else:
		pad_width_left = pad_width_right = pad_width // 2

	image = tf.pad(
		image,
		paddings=[
			[pad_height_top, pad_height_bottom],
			[pad_width_left, pad_width_right],
			[0, 0],
		],
	)

	image = tf.transpose(image, perm=[1, 0, 2])
	image = tf.image.flip_left_right(image)
	return image


#Preprocessing images
batch_size = 64
padding_token = 99
image_width = 128
image_height = 32


def preprocess_image(image_path, img_size=(image_width, image_height)):
	image = tf.io.read_file(image_path)
	image = tf.image.decode_png(image, 1)
	image = distortion_free_resize(image, img_size)
	image = tf.cast(image, tf.float32) / 255.0
	return image


def vectorize_label(label):
	label = char_to_num(tf.strings.unicode_split(
		label, input_encoding="UTF-8"))
	length = tf.shape(label)[0]
	pad_amount = max_len - length
	label = tf.pad(label, paddings=[[0, pad_amount]],
				constant_values=padding_token)
	return label


def process_images_labels(image_path, label):
	image = preprocess_image(image_path)
	label = vectorize_label(label)
	return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
	dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
		process_images_labels, num_parallel_calls=AUTOTUNE
	)
	return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


#Label conversion table and functions
clist = ['5', '6', '8', '9', '0', '1', '4', '-', '7', '2', '3']
AUTOTUNE = tf.data.AUTOTUNE

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=clist, mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
	vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# CTC loss layer (not used but need to prevent model_loading error)


def read_codes(frame):
	# Load the predefined dictionary#Load the dictionary that was used to generate the markers.

	dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

	# Initialize the detector parameters using default values

	parameters =  cv.aruco.DetectorParameters_create()

	# Detect the markers in the image

	#frame = cv.imread("Scan.jpg")
	markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
	#print(markerIds)
	#print(markerCorners)
	#form_id=decode(Image.open('horn4.png'))[0][0].decode("utf-8")
	form_id=decode(frame)[0][0].decode("utf-8")
	#print(form_id)
	return (form_id, markerCorners, markerIds)

def center_aruco(markerCorners, markerIds, pts_src):
	#print(pts_src)
	# Calculate the center of each corner markers
	i = 0
	for each_corner in np.array(markerCorners):
		# print (each_corner)
		# each_corner contains array of four corners coordinates of each aruco marker
		c = np.array(each_corner[0])
		# take out a braket
		x,y = 0,0
		# calculate center point of four corners
		for cor in np.array(c):
			x += cor[0]
			y += cor[1]
			#print("X: ", x, " Y: ",y)
		# xy[markerIds[i][0]] = [x,y]
		# pts_src[markerIds[i][0] - 1] = [x/4,y/4]
		pts_src[i] = [x/4,y/4]
		i += 1
	#print(pts_src)
	return (pts_src)

def read_formdb(form_id):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["forms"]

	myquery = { "form_id": form_id }
	aruco_location = mycol.find(myquery)[0]["aruco"]
	areas = mycol.find(myquery)[0]["areas"]
	layout = mycol.find(myquery)[0]["layout"]
	return(aruco_location, areas, layout)

def set_marker_location(markerIds, aruco_location, pts_dst):
	i = 0; # pts_dst=[None]*len(np.array(markerIds))
	for cor in np.array(markerIds):
		#print(cor)
		#print(aruco_location[str(cor[0])])
		pts_dst[i] = aruco_location[str(cor[0])]
		i += 1
	return (pts_dst)

def sortf(e):
	return e[1]

def eval_logic(img, answer):
	x = np.expand_dims(img, axis=0)
	if answer > 9:
		o_proba = model.predict(x)
		r = 100
	else:
		o_proba = model_single.predict(x)
		r = 10
	o_result = o_proba.tolist()
	pred = int(np.argmax(o_result, axis=-1))
	ave_predict = ans_ave_Pred = o_result[0][1]
	conf_level = 0
	if answer != pred:
		s = o_result
		ns = []
		#print("answer: ",answer,"Number: ",k)
		for i in range(0,r):
			ns.append([i,round(s[0][i]*100,1)])
		ns.sort(reverse=True, key=sortf)
		flag = True
		for l in range(0,5):
			if answer == ns[l][0]:
				flag = False
				print("answer: ",answer," Prediction: ",ns[0], " hit at ",l, ns[l])
				pred = ns[l][0]
				ave_predict = ans_ave_Pred = ns[l][1]
				conf_level = l
				break
		if flag == True:
			print("answer: ",answer, ns[0], ns[1], ns[2], ns[3])
			scan_result=[]
			t1 = False
			sum_cells = 0
			if 9 < answer:
				j1 = int(answer//10)
				j2 = int(answer%10)
				#print ("j1: ", j1, " j2: ", j2)
			for i in range(0,20,3):
				#print(x.shape)
				eex = img[:,i:i+25]
				#print(i, eex.shape)
				ex = np.expand_dims(eex, axis=0)
				s_proba = model_s.predict(ex)
				s_result = s_proba.tolist()
				ind = int(np.argmax(s_result,axis=-1))
				sum_cells += ind
				#print(ind, s_result)
				if 9 < answer:
					if t1 == True and j2 == ind:
						pred = answer
						flag = False
					if t1 == False and j1 == ind:
						t1 = True
				else:
					if ind == answer:
						flag = False
				scan_result.append(ind)
				# scan_result.append([ind, s_result[0][ind]])
			if flag:
				if sum_cells > 60:
					print("answer: ",answer, " Prediction: ",ns[0], "Scan: ", scan_result, "Blank")
					conf_level = -2
				else:
					print("answer: ",answer, " Prediction: ",ns[0], "Scan: ", scan_result)
					conf_level = -1
				ave_predict = ans_ave_Pred = 0
			else:
				print("answer: ",answer, " Prediction: ",ns[0], "Scan: ", scan_result, "hit")
				conf_level = 5
				ave_predict = ans_ave_Pred = 0
	else:
		ave_predict = ans_ave_Pred = o_result[0][0]
		conf_level = 0
	#print("answer: ", answer, "Missed: ", missed, " Within4th: ", within4th, " Scanned: ", scanned)
	return (pred, ave_predict, ans_ave_Pred, conf_level)

def evaluate_img(path, filename, answer):

	#newsize = (45, 25)
	#img = img.resize(newsize)
	#filename = 'F:/Google Drive/User_Backup/test_data/20_1.png'
	print("path: ", path, " filename: ", filename)
	img = image.load_img(path + "/" + filename, color_mode ='grayscale', target_size=(25, 44))
 #   kernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
 #   img = cv.dilate(img, kernel)
 #   img = cv.resize(img,(64, 32), interpolation = cv.INTER_AREA)
 #   img = cv.dilate(img, kernel)
	x = image.img_to_array(img)
	x /= 255
	#x = np.expand_dims(x, axis=0)
	#y_proba = model.predict(x)
	#result = y_proba.tolist()
	test_new_logic = True
	if test_new_logic == True:
		num, ave_predict, ans_ave_pred, conf_level = eval_logic(x, answer)
	else:
		if answer//10 == 0:
			oo_proba = model_oo.predict(x)
			oo_result = oo_proba.tolist()
			ans_ave_pred = oo_result[0][answer]
			ave_predict =oo_result[0][model_o.predict_classes(x)[0]]
			num = [model_oo.predict_classes(x)[0]][0]
		else:
			o_proba = model_o.predict(x)
			t_proba = model_t.predict(x)
			o_result = o_proba.tolist()
			t_result = t_proba.tolist()
			t = [model_t.predict_classes(x)[0]][0]
			o = [model_o.predict_classes(x)[0]][0]
			num = int(t)*10 + int(o)
	#num = int(t)*10 + int(o)
		#ave_predict = (o_result[0][model_o.predict_classes(x)[0]] + t_result[0][model_t.predict_classes(x)[0]])/2
		#ans_ave_pred = (o_result[0][int(answer%10)]+t_result[0][int(answer//10)])/2
			ave_predict = o_result[0][model_o.predict_classes(x)[0]]
			ans_ave_pred = t_result[0][int(answer//10)]
	# Evaluation result
	#print(model.predict_classes(x))
	#for i in range(100):
		#print(i, result[0][i])
	#return (int(model.predict_classes(x)[0]), result[0][model.predict_classes(x)[0]], result[0][answer])
	return (num, ave_predict, ans_ave_pred, conf_level)
	#return (num, .5, .5)


# A utility function to decode the output of the network.
def decode_batch_predictions(pred):
	# Using fixed maximum length 
	max_len = 2

	input_len = np.ones(pred.shape[0]) * pred.shape[1]
	# Use greedy search. For complex tasks, you can use beam search.
	results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
		:, :max_len
	]
	# Iterate over the results and get back the text.
	output_text = []
	for res in results:
		res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
		res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
		output_text.append(res)
	return output_text


def write_evaluation (user, date, image, form_id, evaluation):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["evaluation"]
	result_id = user + date + image[:-4]
	#print(result_id, "Evaluation: ", evaluation)
	key = {"result_id": result_id}
	mydict = {
		"result_id": result_id,
		"user": user,
		"date": date,
		"image": image,
		"form_id": form_id,
		"reviewed": False,
		"mistakes": 0,
		"eval": evaluation}
	mycol.update(key, mydict, upsert = True)
	return

def evaluate_answer(areas, im_out, img, form_id):
	grp = 0
	eval_result=[]; area_result = []; group_result = {} 
	for area in areas:
		row_h = (area["lower_xy"][1] - area["upper_xy"][1])/area["row"]
		col_w = (area["lower_xy"][0] - area["upper_xy"][0])/area["col"]
		group_result["group"] = area["group"]; group_result["row_h"] = row_h; group_result["col_w"] = col_w
		group_result["row"] = area["row"]; group_result["col"] = area["col"]; 
		cell_eval = {}
		#print(upper_xy, lower_xy, row, col)
		batch_images = []; preds = []
		for row in range(area["row"]):
			for col in range(area["col"]):
				upperleft_y = int(area["upper_xy"][1]+row_h*row)
				upperleft_x = int(area["upper_xy"][0]+col_w*col)
				lowerright_y = int(area["upper_xy"][1]+row_h*(row+1))
				lowerright_x = int(area["upper_xy"][0]+col_w*(col+1))
				#print(upperleft_y, lowerright_y, upperleft_x,lowerright_x )
				#Extracging cell image
				#Both left and right side are shorten 1 pixcel to prevent including frame lines
				cut = im_out[upperleft_y:lowerright_y, upperleft_x+1:lowerright_x-1]
				cut = cv.bitwise_not(cut)
				#print(np.shape(cut))
				filename = str(grp)+str(row)+str(col)+".png"
				#path = os.path.join(parent_dir, "uploads", current_user.username, date, img[:-4])
				path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date, img[:-4])
				if os.path.exists(path) == False:
					try:
						os.makedirs(path, exist_ok = True)
						print("Directry '%s' created successfully" %path)
					except OSError as error:
						print("Directory 's%' can not be created")    
				#path = os.path.join(path, filename)    
				cv.imwrite(path + "/" + filename, cut)

				#CNN+RNN model
				cut = np.expand_dims(cut, axis = -1)
				image_tensor = tf.convert_to_tensor(cut, dtype=tf.float32)
				image = distortion_free_resize(image_tensor, (128, 32))
				image = tf.cast(image, tf.float32) / 255.0
				batch_images.append(image)

				#place holders for old interface
				pred = 0; prob_pred = 0; prob_ans = 0; conf_level = 0.5
				#Origina CNN model
				#print(row, col, area["answer"][row][col])
				# Evaluate each cell of image with Keras
				#pred, prob_pred, prob_ans, conf_level = evaluate_img(path, filename, area["answer"][row][col])

				# Prepare for constructing JSON data
				cell_eval["row"] = row; cell_eval["col"] = col
				#cell_eval["upper_x"] = area["upper_xy"][0]; cell_eval["upper_y"] = area["upper_xy"][1]
				cell_eval["upper_x"] = upperleft_x; cell_eval["upper_y"] = upperleft_y; cell_eval["color"]="blue"
				cell_eval["predict"] = pred; cell_eval["prob_pred"] = prob_pred; cell_eval["correct"]=False
				cell_eval["prob_ans"] = prob_ans; cell_eval["conf_level"] = conf_level
				cell_eval["answer"] = area["answer"][row][col]; cell_eval["miss_recog"]= False
				cell_eval["match"] = True
				area_result.append(cell_eval); cell_eval = {} 
		batch_images = tf.convert_to_tensor(batch_images)
		#print(np.shape(batch_images))
		preds = prediction_model.predict(batch_images)
		#print("Preds: ", np.shape(preds))
		pred_texts = decode_batch_predictions(preds)
		for row in range(area["row"]):
			for col in range(area["col"]):
				n = row * area["col"] + col
				if pred_texts[n] != str(area["answer"][row][col]):
					area_result[n]["match"] = False
					area_result[n]["predict"] = pred_texts[n]
		mpath = img[:-4]+"/missed"
		if os.path.exists(mpath) == False:
			try:
				os.makedirs(mpath, exist_ok=True)
				print("Directry '%s' created successfully" % mpath)
			except OSError as error:
				print("Directory 's%' can not be created")
		grp += 1
		#eval_result.append(area_result)
		group_result["eval_cells"] = area_result; area_result = [] 
		eval_result.append(group_result); group_result = {}
	#print (eval_result)
	return (eval_result)

def eval_image(img):
	# read image as gray scale
	#print("img in eval_image ", img)
	frame = cv.imread(img,0)
	# Convert the image to binary iwth cv2.Thresh_OTSU.
	ret2, frame = cv.threshold(frame, 0, 255, cv.THRESH_OTSU)
	form_id, markerCorners, markerIds = read_codes(frame)
	aruco_location, areas, layout = read_formdb(form_id)
	#evaluation = {"user": current_user.username, "image": img, "date": date}
	#print(aruco_location)
	if len(markerIds) > 3:
		pts_src, pts_dst = np.zeros((len(markerIds),2)), np.zeros((len(markerIds),2)) 
		#pts_src = np.array([[0,0],[0,0],[0,0],[0,0]])
		# calculate center of each aruco marker
		#print(pts_src)
		pts_src = center_aruco(markerCorners, markerIds, pts_src)
		#print(pts_src)
		# prepare arrays for homography
		pts_dst = set_marker_location(markerIds, aruco_location, pts_dst)
		# Calculate Homography
		#print("pts_dst: ", pts_dst, "pts_src: ", pts_src)
		h, status = cv.findHomography(pts_src, pts_dst)
		#print(status)
		# Warp source image to destination based on homography
		# final image size (744, 531) needs to be set form by form 
		if layout == "L":
			size = (960, 720)
		else:
			size = (720, 960)
		im_out = cv.warpPerspective(frame, h, size)
		#path = os.path.join(parent_dir, "uploads", current_user.username, date, img[:-4])
		path = img[:-4]
		#print("makedir path: ", path)
		if os.path.exists(path) == False:
			try:
				os.makedirs(path, exist_ok = True)
				print("Directry '%s' created successfully" %path)
			except OSError as error:
				print("Directory 's%' can not be created") 
		path = os.path.join(img[:-4], "adjust.jpg")
		#print("Eval_image, imwrite: ", path)
		x = cv.imwrite(path, im_out)
		#print(x)
		eval_result = evaluate_answer(areas, im_out, img, form_id)
		#evaluation["evaluation"] = eval_result
	else:
		print("Error: program is not able to detect four or more corner marks")
	#return (evaluation, path) 
	return (eval_result, path, form_id) 

@app.route("/upload_files", methods=['GET', 'POST'])
@login_required
def upload_files():
	if current_user.is_authenticated == False:
		flash('Login required. Please login before uploading files', 'danger')
		return redirect(url_for('login'))
	form = UploadForm()
	global files, path, secure_files, date, form_id
	secure_files = []
	if form.validate_on_submit():
		files = form.file.data
		if date == "2020-01-01":
			return redirect(url_for('testdate'))
		for file in files:
			if allowed_file(file.filename) == False:
				msg = "Wrong file format: " + file.filename
				flash(msg) 
			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)
				#path = os.path.join(parent_dir, "uploads", current_user.username, date)
				path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date)
				file.save(os.path.join(path, filename))
				secure_files.append(filename)
				#msg = "Uploaded: " + filename
				#flash(msg)
		#print("Secure_Files: ", secure_files)
		for file in secure_files:
			print("Upload_files' file", file)
			#path = os.path.join(parent_dir, "uploads", current_user.username, date)
			path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date)
			eval_result, path, form_id = eval_image(os.path.join(path, file))
			write_evaluation (current_user.username, date, file, form_id, eval_result)
		#    eval_result = eval_image(img)
		return render_template("public/uploaded.html", title="Uploaded", form = form, files = secure_files)
	elif request.method == 'GET':
		return render_template("public/upload_files.html", title="Upload", form = form)
	else:
		flash('Else')
		return render_template("public/upload_files.html", title="Upload", form = form)

def read_result (user, date, image):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["evaluation"]
	result_id = user + date + image[:-4]
	myquery = {"result_id": result_id}
	mydoc = mycol.find(myquery,{"_id":0, "result_id": 1, "user": 1, "date": 1, "image": 1, "form_id": 1, "eval": 1,"reviewed": 1, "mistakes": 1})
	evaluation = {}
	for x in mydoc:
		evaluation = x
		continue
	return evaluation

def update_result (user, date, image, form_id, data):
	#global sid
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["evaluation"]
	data["timestamp"] = datetime.now()
	data["pg"]="Batch"
	result_id = user + date + image[:-4]
	#result_id = user + str(sid) + date + image[:-4]
	myquery = {"result_id": result_id}
	#newvalues = { "$set" : data}
	newvalues = {
		"$set": {"mistakes": data["mistakes"], "reviewed": True, "eval": data["eval"]}}
	mycol.update_one(myquery, newvalues)
	path = os.path.join(app.root_path, 'static/img/upload/', data["user"], data["date"], data["image"][:-4] )
	#print("Update Pymongo path: ", path)
	#print("data: ", data)
	for row in range(int(data["eval"][0]["row"])):
		for col in range(int(data["eval"][0]["col"])):
			n = row * data["eval"][0]["col"] + col
			if (data["eval"][0]["eval_cells"][n]["miss_recog"] == True):
				#print("row: ", data["eval"][0]["row"], "col: ", data["eval"][0]["col"], "n: ", n,
                #            "match: ", data["eval"][0]["eval_cells"][n]["match"], "correct: ", data["eval"][0]["eval_cells"][n]["correct"])
				mpath = path + "/missed"
				if os.path.exists(mpath) == False:
					try:
						os.makedirs(mpath, exist_ok=True)
						print("Directry '%s' created successfully" % mpath)
					except OSError as error:
						print("Directory 's%' can not be created")
				filename = "0"+str(row)+str(col)+".png"
					# path = os.path.join(app.root_path, 'static/img/upload/',
					#                     current_user.username, date, img[:-4])
				cut = cv.imread(path + "/" + filename, 0)
					# if area["answer"][row][col] >= 10:
					# 	filename = str(row*area["row"] + col) + "-" + str(area["answer"][row][col]) +".png"
					# else:
					# 	filename = str(row*area["row"] + col) + "-" + "0" + str(area["answer"][row][col]) +".png"
				cv.imwrite(mpath + "/" + filename, cut)
				#print(grp, row, col, pred, prob_pred, prob_ans, area["answer"][row][col])
	return 

def write_review_log (user, date, image, form_id, data):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["review_log"]
	data["timestamp"] = datetime.now()
	data["pg"] = "Batch"
	result_id = user + date + image[:-4]
	key = {"result_id": result_id}
	mydict = data
	#print(data)
	mycol.update(key, mydict, upsert = True)
	return

def return_page (secure_files, i):
	#print("i in line 659: ", i)
	#print("secure_files: ", secure_files)
	evaluation = read_result(current_user.username, date, secure_files[i] )
	#print(evaluation)
	#path = os.path.join(parent_dir, "uploads", current_user.username, date)
	path = os.path.join('static\\img\\upload\\', current_user.username, date, secure_files[i][:-4],"adjust.jpg" )
	return_json = {
			'path': path,
			'total_files': len(secure_files),
			'number_of_file':i,
			'file': secure_files[i],
			'eval': evaluation,
			'type': "eval"
		}
	#with open ("C:/Users/Hiroji/Documents/json.txt","w") as json_file:
	#    json.dump(return_json, json_file)
	res = make_response(jsonify(return_json), 200) 
	return res

def return_endpage (secure_files):
	res = []
	for f in secure_files:  
		evaluation = read_result(current_user.username, date, f )
		res.append({"image": evaluation["image"],"mistakes": evaluation["mistakes"] })
	return_json = {
			'path': "",
			'total_files': len(secure_files),
			'number_of_file':i,
			'file': secure_files[i],
			'eval': res,
			'type': "end"
		}
	#with open ("C:/Users/Hiroji/Documents/json.txt","w") as json_file:
	#    json.dump(return_json, json_file)
	res = make_response(jsonify(return_json), 200) 
	return res


@app.route("/api", methods=['POST'])
def api():
	global secure_files, path, i

	req = request.get_json()  
	print("Request: ", req["action"], "secure_files in api(): ", secure_files)
	if req["action"] == "JSON":
		i = 0
		res = return_page (secure_files, i)
	else:
		write_review_log(current_user.username, date, secure_files[i], req["data"]["form_id"], req["data"])
		update_result(current_user.username, date, secure_files[i], req["data"]["form_id"], req["data"])
		if req["action"] == "End":
			i = 0
			res = return_endpage (secure_files)
		elif req["action"] == "+" and i <len(secure_files):
			i += 1
			res = return_page (secure_files, i)
		elif req["action"] == "-" and i > 0:
			i -= 1
			res = return_page (secure_files, i)
	return res        

## @app.route("/test", methods=['GET', 'POST'])
## def test():
##    if request.method == "POST":
##        firstname = request.form['firstname']
##        lastname = request.form['lastname']


@app.route("/terms", methods=['GET'])
def terms():
	return render_template("public/terms.html")


@app.route("/privacy", methods=['GET'])
def privacy():
	return render_template("public/privacy.html")
