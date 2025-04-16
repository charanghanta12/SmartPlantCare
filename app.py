from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from pymongo.errors import PyMongoError
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask_mail import Mail, Message
import secrets
import socket
from flask import Response, abort
import gridfs

# Load environment variables
load_dotenv()

# Create Flask app with proper configuration
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key_here')

# Configure debug mode carefully
app.config['DEBUG'] = os.getenv('DEBUG', 'False') == 'True'
app.config['USE_RELOADER'] = False  # Disable reloader to prevent threading issues

# Configure MongoDB
app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/SmartPlantCare")
mongo = PyMongo(app)

fs = gridfs.GridFS(mongo.db)

# Email Configuration
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USE_SSL=False,
    MAIL_USERNAME=os.getenv('MAIL_USERNAME'),
    MAIL_PASSWORD=os.getenv('MAIL_PASSWORD'),
    MAIL_DEFAULT_SENDER=(os.getenv('MAIL_USERNAME', 'noreply@smartplantcare.com'), 'Smart Plant Care'),
    MAIL_DEBUG=app.debug,
    MAIL_SUPPRESS_SEND=False,
    MAIL_TIMEOUT=10
)


# Initialize Mail after app configuration
mail = Mail(app)

# Load ML Models
try:
    plant_disease_model = tf.keras.models.load_model("models/plant_disease_model.keras")
    plant_watering_model = joblib.load("models/plant_watering_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disease class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Initialize MongoDB collections and indexes
with app.app_context():
    try:
        mongo.db.users.create_index("email", unique=True)
        mongo.db.plants.create_index("userId")
        mongo.db.predictions.create_index([("userId", 1), ("timestamp", -1)])
        mongo.db.password_resets.create_index("token", unique=True)
        mongo.db.password_resets.create_index("created_at", expireAfterSeconds=3600)  # 1 hour expiration
    except Exception as e:
        print(f"Error creating MongoDB indexes: {e}")

# Helper Functions
def model_prediction(image_path):
    try:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        predictions = plant_disease_model.predict(input_arr)
        return np.argmax(predictions)
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return None

def is_token_valid(token):
    """Check if token exists and hasn't expired"""
    try:
        reset_entry = mongo.db.password_resets.find_one({'token': token})
        if not reset_entry:
            return False
        return datetime.utcnow() - reset_entry['created_at'] <= timedelta(hours=1)
    except Exception as e:
        print(f"Error checking token validity: {e}")
        return False

# Authentication Routes
@app.route('/api/auth/status', methods=['GET'])
def check_login_status():
    if 'user_id' in session:
        try:
            user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])}, {'password': 0})
            if user:
                return jsonify({
                    'isLoggedIn': True,
                    'user': {
                        'id': str(user['_id']),
                        'name': user['name'],
                        'email': user['email']
                    }
                }), 200
        except Exception as e:
            return jsonify({'message': 'Database error', 'error': str(e)}), 500
    return jsonify({'isLoggedIn': False}), 200

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not all(key in data for key in ['name', 'email', 'password']):
        return jsonify({'message': 'Missing required fields'}), 400

    try:
        if mongo.db.users.find_one({'email': data['email']}):
            return jsonify({'message': 'Email already registered'}), 400

        user_data = {
            'name': data['name'],
            'email': data['email'],
            'password': generate_password_hash(data['password']),
            'createdAt': datetime.utcnow(),
            'updatedAt': datetime.utcnow()
        }

        result = mongo.db.users.insert_one(user_data)
        user_id = str(result.inserted_id)

        session['user_id'] = user_id
        session['user_name'] = data['name']

        return jsonify({
            'message': 'Registration successful',
            'user': {
                'id': user_id,
                'name': data['name'],
                'email': data['email']
            }
        }), 201
    except Exception as e:
        return jsonify({'message': 'Registration failed', 'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not all(key in data for key in ['email', 'password']):
        return jsonify({'message': 'Missing email or password'}), 400

    try:
        user = mongo.db.users.find_one({'email': data['email']})
        if user and check_password_hash(user['password'], data['password']):
            session['user_id'] = str(user['_id'])
            session['user_name'] = user['name']

            return jsonify({
                'message': 'Login successful',
                'user': {
                    'id': str(user['_id']),
                    'name': user['name'],
                    'email': user['email']
                }
            }), 200
        return jsonify({'message': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'message': 'Database error', 'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({'message': 'Email is required'}), 400

    try:
        user = mongo.db.users.find_one({'email': email})
        if not user:
            # Return same message whether user exists or not for security
            return jsonify({'message': 'If this email exists, a reset link has been sent'}), 200

        # Delete any existing reset tokens for this email
        mongo.db.password_resets.delete_many({'email': email})

        reset_token = secrets.token_urlsafe(32)
        mongo.db.password_resets.insert_one({
            'email': email,
            'token': reset_token,
            'created_at': datetime.utcnow()
        })

        reset_link = f"{request.host_url}reset-password?token={reset_token}"

        try:
            msg = Message(
                "Password Reset Request",
                recipients=[email],
                html=f"""
                <h2>Password Reset Request</h2>
                <p>Click the button below to reset your password:</p>
                <a href="{reset_link}" style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">Reset Password</a>
                <p>This link will expire in 1 hour.</p>
                <p>If you didn't request this, please ignore this email.</p>
                """
            )
            mail.send(msg)
            return jsonify({'message': 'If this email exists, a reset link has been sent'}), 200
        except Exception as e:
            app.logger.error(f"Failed to send email: {str(e)}")
            return jsonify({'message': 'Failed to send reset email'}), 500

    except Exception as e:
        app.logger.error(f"Error in forgot password: {str(e)}")
        return jsonify({'message': 'An error occurred'}), 500
@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password_page():
    token = request.args.get('token') if request.method == 'GET' else request.form.get('token')

    if not token or not is_token_valid(token):
        flash('Invalid or expired reset link', 'error')
        return redirect(url_for('home'))

    if request.method == 'POST':
        new_password = request.form.get('newPassword')
        confirm_password = request.form.get('confirmPassword')

        if new_password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('reset_password.html', token=token)

        try:
            reset_entry = mongo.db.password_resets.find_one({'token': token})
            if not reset_entry:
                flash('Invalid or expired token', 'error')
                return redirect(url_for('home'))

            hashed_password = generate_password_hash(new_password)
            mongo.db.users.update_one(
                {'email': reset_entry['email']},
                {'$set': {'password': hashed_password, 'updatedAt': datetime.utcnow()}}
            )

            mongo.db.password_resets.delete_one({'token': token})
            flash('Password reset successfully! Please login with your new password.', 'success')
            return redirect(url_for('home'))

        except Exception as e:
            flash('An error occurred. Please try again.', 'error')
            return render_template('reset_password.html', token=token)

    return render_template('reset_password.html', token=token)

# Plant Management Routes
@app.route('/api/plants', methods=['POST'])
def add_plant():
    if 'user_id' not in session:
        return jsonify({'message': 'Unauthorized'}), 401

    data = request.get_json()
    required_fields = ['name', 'type', 'plantingDate']
    if not data or not all(field in data for field in required_fields):
        return jsonify({'message': 'Missing required fields'}), 400

    try:
        plant_data = {
            'userId': ObjectId(session['user_id']),
            'name': data['name'],
            'type': data['type'],
            'plantingDate': data['plantingDate'],
            'lastWatered': data.get('lastWatered', datetime.utcnow()),
            'healthStatus': data.get('healthStatus', 'healthy'),
            'notes': data.get('notes', ''),
            'createdAt': datetime.utcnow(),
            'updatedAt': datetime.utcnow()
        }

        result = mongo.db.plants.insert_one(plant_data)
        return jsonify({
            'message': 'Plant added successfully',
            'plantId': str(result.inserted_id)
        }), 201
    except Exception as e:
        return jsonify({'message': 'Failed to add plant', 'error': str(e)}), 500

@app.route('/api/plants', methods=['GET'])
def get_plants():
    if 'user_id' not in session:
        return jsonify({'message': 'Unauthorized'}), 401

    try:
        plants = list(mongo.db.plants.find({'userId': ObjectId(session['user_id'])}))
        for plant in plants:
            plant['_id'] = str(plant['_id'])
            plant['userId'] = str(plant['userId'])
        return jsonify(plants), 200
    except Exception as e:
        return jsonify({'message': 'Failed to fetch plants', 'error': str(e)}), 500

@app.route('/api/plants/<plant_id>', methods=['PUT'])
def update_plant(plant_id):
    if 'user_id' not in session:
        return jsonify({'message': 'Unauthorized'}), 401

    data = request.get_json()
    if not data:
        return jsonify({'message': 'No data provided'}), 400

    try:
        updates = {
            'updatedAt': datetime.utcnow()
        }

        if 'name' in data:
            updates['name'] = data['name']
        if 'type' in data:
            updates['type'] = data['type']
        if 'healthStatus' in data:
            updates['healthStatus'] = data['healthStatus']
        if 'notes' in data:
            updates['notes'] = data['notes']
        if 'lastWatered' in data:
            updates['lastWatered'] = data['lastWatered']

        result = mongo.db.plants.update_one(
            {'_id': ObjectId(plant_id), 'userId': ObjectId(session['user_id'])},
            {'$set': updates}
        )
        if result.modified_count == 0:
            return jsonify({'message': 'Plant not found or no changes made'}), 404
        return jsonify({'message': 'Plant updated successfully'}), 200
    except Exception as e:
        return jsonify({'message': 'Failed to update plant', 'error': str(e)}), 500

# Prediction Routes
@app.route('/api/predict_disease', methods=['GET', 'POST'])
def predict_disease_api():
    if 'user_id' not in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('disease.html', error='No file uploaded')

        file = request.files['image']
        if file.filename == '':
            return render_template('disease.html', error='No file selected')

        try:
            # Save image to MongoDB GridFS
            image_id = fs.put(file.read(), filename=file.filename, content_type=file.content_type)

            # Temporarily save image locally for prediction
            temp_path = f"temp_{datetime.now().timestamp()}.jpg"
            with open(temp_path, 'wb') as f:
                f.write(fs.get(image_id).read())

            # Predict disease
            result_index = model_prediction(temp_path)
            os.remove(temp_path)

            prediction = class_names[result_index]
            plant_type, condition = prediction.split('___')
            formatted_prediction = f"{plant_type.replace('_', ' ')}: {condition.replace('_', ' ')}"

            # Save prediction in MongoDB
            mongo.db.predictions.insert_one({
                'userId': ObjectId(session['user_id']),
                'type': 'disease',
                'result': formatted_prediction,
                'imageId': image_id,
                'timestamp': datetime.utcnow()
            })

            return render_template('disease.html', prediction=formatted_prediction, image_id=str(image_id))

        except Exception as e:
            return render_template('disease.html', error=f"Error: {str(e)}")

    return render_template('disease.html')


@app.route('/api/predict/watering', methods=['POST'])
def predict_watering_api():
    if 'user_id' not in session:
        return jsonify({'message': 'Unauthorized'}), 401

    data = request.get_json()
    required_fields = [
        'light_intensity', 'temperature', 'humidity', 'soil_moisture',
        'soil_temperature', 'soil_ph', 'soil_ec', 'leaf_temperature',
        'atmospheric_pressure', 'vapor_density', 'heat_index',
        'rain_status', 'cloud_status', 'wind_status'
    ]

    if not data or not all(field in data for field in required_fields):
        return jsonify({'message': 'Missing required fields'}), 400

    try:
        input_data = [float(data[field]) for field in required_fields]
        input_arr = np.asarray(input_data).reshape(1, -1)
        std_data = scaler.transform(input_arr)
        prediction = plant_watering_model.predict(std_data)[0]

        recommendation = "Your plant needs watering!" if prediction == 1 else "Your plant does not need watering now."
        status = "needs_watering" if prediction == 1 else "no_watering"

        mongo.db.predictions.insert_one({
            'userId': ObjectId(session['user_id']),
            'type': 'watering',
            'result': recommendation,
            'inputData': {field: data[field] for field in required_fields},
            'timestamp': datetime.utcnow()
        })

        return jsonify({
            'recommendation': recommendation,
            'status': status
        }), 200
    except Exception as e:
        return jsonify({'message': f"Error: {str(e)}"}), 500

@app.route('/image/<image_id>')
def get_image(image_id):
    try:
        image = fs.get(ObjectId(image_id))
        return Response(image.read(), mimetype=image.content_type)
    except:
        abort(404)

# Frontend Routes
@app.route('/')
def home():
    is_logged_in = 'user_id' in session
    user_name = session.get('user_name', '')
    return render_template('index.html', is_logged_in=is_logged_in, user_name=user_name)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('home'))

    try:
        plants = list(mongo.db.plants.find(
            {'userId': ObjectId(session['user_id'])},
            {'_id': 1, 'name': 1, 'type': 1, 'healthStatus': 1}
        ).limit(5))

        predictions = list(mongo.db.predictions.find(
            {'userId': ObjectId(session['user_id'])},
            {'_id': 0, 'type': 1, 'result': 1, 'timestamp': 1}
        ).sort('timestamp', -1).limit(5))

        for plant in plants:
            plant['_id'] = str(plant['_id'])

        return render_template('dashboard.html',
                               user_name=session.get('user_name', ''),
                               plants=plants,
                               predictions=predictions,
                               is_logged_in=True)
    except Exception as e:
        flash('Error loading dashboard data')
        return render_template('dashboard.html',
                               user_name=session.get('user_name', ''),
                               plants=[],
                               predictions=[],
                               is_logged_in=True)

@app.route('/predict_disease')
def predict_disease():
    is_logged_in = 'user_id' in session
    return render_template('disease.html', is_logged_in=is_logged_in)

@app.route('/predict_watering')
def predict_watering():
    is_logged_in = 'user_id' in session
    return render_template('watering.html', is_logged_in=is_logged_in)

@app.route('/my_plants')
def my_plants():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('my_plants.html', is_logged_in=True)

@app.route('/community')
def community():
    is_logged_in = 'user_id' in session
    return render_template('community.html', is_logged_in=is_logged_in)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('profile.html', is_logged_in=True)


@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('home'))

    try:
        history = list(mongo.db.predictions.find({'userId': ObjectId(session['user_id'])}).sort('timestamp', -1))
        for item in history:
            item['_id'] = str(item['_id'])
            item['userId'] = str(item['userId'])
        return render_template('history.html', history=history, is_logged_in=True)
    except Exception as e:
        flash("Error loading history")
        return render_template('history.html', history=[], is_logged_in=True)


@app.route('/settings')
def settings():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('settings.html', is_logged_in=True)

if __name__ == '__main__':
    try:
        port = int(os.getenv('PORT', 5000))
        use_debugger = os.getenv('USE_DEBUGGER', 'true').lower() == 'true'

        from werkzeug.serving import run_simple

        run_simple(
            'localhost',
            port,
            app,
            use_debugger=use_debugger,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        raise


#     scaler = None
#
# # Disease class names
# class_names = [
#     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#     'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
#     'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#     'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
#     'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#     'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
#     'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
#     'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
#     'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
#     'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
#     'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
#     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
#     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#     'Tomato___healthy'
# ]
#
# # Initialize MongoDB collections and indexes
# with app.app_context():
#     try:
#         mongo.db.users.create_index("email", unique=True)
#         mongo.db.plants.create_index("userId")
#         mongo.db.predictions.create_index([("userId", 1), ("timestamp", -1)])
#         mongo.db.password_resets.create_index("token", unique=True)
#         mongo.db.password_resets.create_index("created_at", expireAfterSeconds=3600)  # 1 hour expiration
#     except Exception as e:
#         print(f"Error creating MongoDB indexes: {e}")
#
#
# # Helper Functions
# def model_prediction(image_path):
#     if plant_disease_model is None:
#         print("Model not loaded correctly")
#         return None
#
#     try:
#         if not os.path.exists(image_path):
#             print(f"Image file not found: {image_path}")
#             return None
#
#         image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
#         input_arr = tf.keras.preprocessing.image.img_to_array(image)
#         input_arr = np.array([input_arr]) / 255.0  # Normalize the image data
#
#         predictions = plant_disease_model.predict(input_arr)
#         return np.argmax(predictions)
#     except Exception as e:
#         print(f"Error in model prediction: {e}")
#         return None
#
#
# def is_token_valid(token):
#     """Check if token exists and hasn't expired"""
#     try:
#         reset_entry = mongo.db.password_resets.find_one({'token': token})
#         if not reset_entry:
#             return False
#         return datetime.utcnow() - reset_entry['created_at'] <= timedelta(hours=1)
#     except Exception as e:
#         print(f"Error checking token validity: {e}")
#         return False
#
#
# # Authentication Routes
# @app.route('/api/auth/status', methods=['GET'])
# def check_login_status():
#     if 'user_id' in session:
#         try:
#             user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])}, {'password': 0})
#             if user:
#                 return jsonify({
#                     'isLoggedIn': True,
#                     'user': {
#                         'id': str(user['_id']),
#                         'name': user['name'],
#                         'email': user['email']
#                     }
#                 }), 200
#         except Exception as e:
#             return jsonify({'message': 'Database error', 'error': str(e)}), 500
#     return jsonify({'isLoggedIn': False}), 200
#
#
# @app.route('/api/auth/register', methods=['POST'])
# def register():
#     data = request.get_json()
#     if not data or not all(key in data for key in ['name', 'email', 'password']):
#         return jsonify({'message': 'Missing required fields'}), 400
#
#     try:
#         if mongo.db.users.find_one({'email': data['email']}):
#             return jsonify({'message': 'Email already registered'}), 400
#
#         user_data = {
#             'name': data['name'],
#             'email': data['email'],
#             'password': generate_password_hash(data['password']),
#             'createdAt': datetime.utcnow(),
#             'updatedAt': datetime.utcnow()
#         }
#
#         result = mongo.db.users.insert_one(user_data)
#         user_id = str(result.inserted_id)
#
#         session['user_id'] = user_id
#         session['user_name'] = data['name']
#
#         return jsonify({
#             'message': 'Registration successful',
#             'user': {
#                 'id': user_id,
#                 'name': data['name'],
#                 'email': data['email']
#             }
#         }), 201
#     except Exception as e:
#         return jsonify({'message': 'Registration failed', 'error': str(e)}), 500
#
#
# @app.route('/api/auth/login', methods=['POST'])
# def login():
#     data = request.get_json()
#     if not data or not all(key in data for key in ['email', 'password']):
#         return jsonify({'message': 'Missing email or password'}), 400
#
#     try:
#         user = mongo.db.users.find_one({'email': data['email']})
#         if user and check_password_hash(user['password'], data['password']):
#             session['user_id'] = str(user['_id'])
#             session['user_name'] = user['name']
#
#             return jsonify({
#                 'message': 'Login successful',
#                 'user': {
#                     'id': str(user['_id']),
#                     'name': user['name'],
#                     'email': user['email']
#                 }
#             }), 200
#         return jsonify({'message': 'Invalid credentials'}), 401
#     except Exception as e:
#         return jsonify({'message': 'Database error', 'error': str(e)}), 500
#
#
# @app.route('/api/auth/logout', methods=['POST'])
# def logout():
#     session.clear()
#     return jsonify({'message': 'Logged out successfully'}), 200
#
#
# @app.route('/api/auth/forgot-password', methods=['POST'])
# def forgot_password():
#     data = request.get_json()
#     email = data.get('email')
#
#     if not email:
#         return jsonify({'message': 'Email is required'}), 400
#
#     try:
#         user = mongo.db.users.find_one({'email': email})
#         if not user:
#             # Return same message whether user exists or not for security
#             return jsonify({'message': 'If this email exists, a reset link has been sent'}), 200
#
#         # Delete any existing reset tokens for this email
#         mongo.db.password_resets.delete_many({'email': email})
#
#         reset_token = secrets.token_urlsafe(32)
#         mongo.db.password_resets.insert_one({
#             'email': email,
#             'token': reset_token,
#             'created_at': datetime.utcnow()
#         })
#
#         reset_link = f"{request.host_url}reset-password?token={reset_token}"
#
#         try:
#             msg = Message(
#                 "Password Reset Request",
#                 recipients=[email],
#                 html=f"""
#                 <h2>Password Reset Request</h2>
#                 <p>Click the button below to reset your password:</p>
#                 <a href="{reset_link}" style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">Reset Password</a>
#                 <p>This link will expire in 1 hour.</p>
#                 <p>If you didn't request this, please ignore this email.</p>
#                 """
#             )
#             mail.send(msg)
#             return jsonify({'message': 'If this email exists, a reset link has been sent'}), 200
#         except Exception as e:
#             app.logger.error(f"Failed to send email: {str(e)}")
#             return jsonify({'message': 'Failed to send reset email'}), 500
#
#     except Exception as e:
#         app.logger.error(f"Error in forgot password: {str(e)}")
#         return jsonify({'message': 'An error occurred'}), 500
#
#
# @app.route('/reset-password', methods=['GET', 'POST'])
# def reset_password_page():
#     token = request.args.get('token') if request.method == 'GET' else request.form.get('token')
#
#     if not token or not is_token_valid(token):
#         flash('Invalid or expired reset link', 'error')
#         return redirect(url_for('home'))
#
#     if request.method == 'POST':
#         new_password = request.form.get('newPassword')
#         confirm_password = request.form.get('confirmPassword')
#
#         if new_password != confirm_password:
#             flash('Passwords do not match', 'error')
#             return render_template('reset_password.html', token=token)
#
#         try:
#             reset_entry = mongo.db.password_resets.find_one({'token': token})
#             if not reset_entry:
#                 flash('Invalid or expired token', 'error')
#                 return redirect(url_for('home'))
#
#             hashed_password = generate_password_hash(new_password)
#             mongo.db.users.update_one(
#                 {'email': reset_entry['email']},
#                 {'$set': {'password': hashed_password, 'updatedAt': datetime.utcnow()}}
#             )
#
#             mongo.db.password_resets.delete_one({'token': token})
#             flash('Password reset successfully! Please login with your new password.', 'success')
#             return redirect(url_for('home'))
#
#         except Exception as e:
#             flash('An error occurred. Please try again.', 'error')
#             return render_template('reset_password.html', token=token)
#
#     return render_template('reset_password.html', token=token)
#
#
# # Plant Management Routes
# @app.route('/api/plants', methods=['POST'])
# def add_plant():
#     if 'user_id' not in session:
#         return jsonify({'message': 'Unauthorized'}), 401
#
#     data = request.get_json()
#     required_fields = ['name', 'type', 'plantingDate']
#     if not data or not all(field in data for field in required_fields):
#         return jsonify({'message': 'Missing required fields'}), 400
#
#     try:
#         plant_data = {
#             'userId': ObjectId(session['user_id']),
#             'name': data['name'],
#             'type': data['type'],
#             'plantingDate': data['plantingDate'],
#             'lastWatered': data.get('lastWatered', datetime.utcnow()),
#             'healthStatus': data.get('healthStatus', 'healthy'),
#             'notes': data.get('notes', ''),
#             'createdAt': datetime.utcnow(),
#             'updatedAt': datetime.utcnow()
#         }
#
#         result = mongo.db.plants.insert_one(plant_data)
#         return jsonify({
#             'message': 'Plant added successfully',
#             'plantId': str(result.inserted_id)
#         }), 201
#     except Exception as e:
#         return jsonify({'message': 'Failed to add plant', 'error': str(e)}), 500
#
#
# @app.route('/api/plants', methods=['GET'])
# def get_plants():
#     if 'user_id' not in session:
#         return jsonify({'message': 'Unauthorized'}), 401
#
#     try:
#         plants = list(mongo.db.plants.find({'userId': ObjectId(session['user_id'])}))
#         for plant in plants:
#             plant['_id'] = str(plant['_id'])
#             plant['userId'] = str(plant['userId'])
#         return jsonify(plants), 200
#     except Exception as e:
#         return jsonify({'message': 'Failed to fetch plants', 'error': str(e)}), 500
#
#
# @app.route('/api/plants/<plant_id>', methods=['PUT'])
# def update_plant(plant_id):
#     if 'user_id' not in session:
#         return jsonify({'message': 'Unauthorized'}), 401
#
#     data = request.get_json()
#     if not data:
#         return jsonify({'message': 'No data provided'}), 400
#
#     try:
#         updates = {
#             'updatedAt': datetime.utcnow()
#         }
#
#         if 'name' in data:
#             updates['name'] = data['name']
#         if 'type' in data:
#             updates['type'] = data['type']
#         if 'healthStatus' in data:
#             updates['healthStatus'] = data['healthStatus']
#         if 'notes' in data:
#             updates['notes'] = data['notes']
#         if 'lastWatered' in data:
#             updates['lastWatered'] = data['lastWatered']
#
#         result = mongo.db.plants.update_one(
#             {'_id': ObjectId(plant_id), 'userId': ObjectId(session['user_id'])},
#             {'$set': updates}
#         )
#         if result.modified_count == 0:
#             return jsonify({'message': 'Plant not found or no changes made'}), 404
#         return jsonify({'message': 'Plant updated successfully'}), 200
#     except Exception as e:
#         return jsonify({'message': 'Failed to update plant', 'error': str(e)}), 500
#
#
# # Prediction Routes
# @app.route('/api/predict_disease', methods=['GET', 'POST'])
# def predict_disease_api():
#     if 'user_id' not in session:
#         return redirect(url_for('home'))
#
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return render_template('disease.html', error='No file uploaded')
#
#         file = request.files['image']
#         if file.filename == '':
#             return render_template('disease.html', error='No file selected')
#
#         try:
#             # Ensure upload folder exists
#             os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#
#             # Save image to MongoDB GridFS
#             file_content = file.read()
#             image_id = fs.put(file_content, filename=file.filename, content_type=file.content_type)
#
#             # Temporarily save image locally for prediction
#             temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{datetime.now().timestamp()}.jpg")
#             with open(temp_path, 'wb') as f:
#                 f.write(fs.get(image_id).read())
#
#             # Predict disease
#             result_index = model_prediction(temp_path)
#
#             # Clean up temporary file
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)
#
#             if result_index is None:
#                 return render_template('disease.html',
#                                        error="Error processing image. Please try again with a clearer image.")
#
#             prediction = class_names[result_index]
#             plant_type, condition = prediction.split('___')
#             formatted_prediction = f"{plant_type.replace('_', ' ')}: {condition.replace('_', ' ')}"
#
#             # Save prediction in MongoDB
#             mongo.db.predictions.insert_one({
#                 'userId': ObjectId(session['user_id']),
#                 'type': 'disease',
#                 'result': formatted_prediction,
#                 'imageId': image_id,
#                 'timestamp': datetime.utcnow()
#             })
#
#             return render_template('disease.html', prediction=formatted_prediction, image_id=str(image_id))
#
#         except Exception as e:
#             app.logger.error(f"Error in disease prediction: {str(e)}")
#             return render_template('disease.html', error=f"Error: {str(e)}")
#
#     return render_template('disease.html')
#
#
# @app.route('/api/predict/watering', methods=['POST'])
# def predict_watering_api():
#     if 'user_id' not in session:
#         return jsonify({'message': 'Unauthorized'}), 401
#
#     if plant_watering_model is None or scaler is None:
#         return jsonify({'message': 'Models not loaded correctly'}), 500
#
#     data = request.get_json()
#     required_fields = [
#         'light_intensity', 'temperature', 'humidity', 'soil_moisture',
#         'soil_temperature', 'soil_ph', 'soil_ec', 'leaf_temperature',
#         'atmospheric_pressure', 'vapor_density', 'heat_index',
#         'rain_status', 'cloud_status', 'wind_status'
#     ]
#
#     if not data or not all(field in data for field in required_fields):
#         return jsonify({'message': 'Missing required fields'}), 400
#
#     try:
#         input_data = [float(data[field]) for field in required_fields]
#         input_arr = np.asarray(input_data).reshape(1, -1)
#         std_data = scaler.transform(input_arr)
#         prediction = plant_watering_model.predict(std_data)[0]
#
#         recommendation = "Your plant needs watering!" if prediction == 1 else "Your plant does not need watering now."
#         status = "needs_watering" if prediction == 1 else "no_watering"
#
#         mongo.db.predictions.insert_one({
#             'userId': ObjectId(session['user_id']),
#             'type': 'watering',
#             'result': recommendation,
#             'inputData': {field: data[field] for field in required_fields},
#             'timestamp': datetime.utcnow()
#         })
#
#         return jsonify({
#             'recommendation': recommendation,
#             'status': status
#         }), 200
#     except Exception as e:
#         return jsonify({'message': f"Error: {str(e)}"}), 500
#
#
# @app.route('/image/<image_id>')
# def get_image(image_id):
#     try:
#         if not ObjectId.is_valid(image_id):
#             return "Invalid image ID", 400
#
#         image = fs.get(ObjectId(image_id))
#         if not image:
#             return "Image not found", 404
#
#         return Response(image.read(), mimetype=image.content_type)
#     except Exception as e:
#         app.logger.error(f"Error retrieving image: {e}")
#         return "Image not found", 404
#
#
# # Frontend Routes
# @app.route('/')
# def home():
#     is_logged_in = 'user_id' in session
#     user_name = session.get('user_name', '')
#     return render_template('index.html', is_logged_in=is_logged_in, user_name=user_name)
#
#
# @app.route('/dashboard')
# def dashboard():
#     if 'user_id' not in session:
#         return redirect(url_for('home'))
#
#     try:
#         plants = list(mongo.db.plants.find(
#             {'userId': ObjectId(session['user_id'])},
#             {'_id': 1, 'name': 1, 'type': 1, 'healthStatus': 1}
#         ).limit(5))
#
#         predictions = list(mongo.db.predictions.find(
#             {'userId': ObjectId(session['user_id'])},
#             {'_id': 0, 'type': 1, 'result': 1, 'timestamp': 1}
#         ).sort('timestamp', -1).limit(5))
#
#         for plant in plants:
#             plant['_id'] = str(plant['_id'])
#
#         return render_template('dashboard.html',
#                                user_name=session.get('user_name', ''),
#                                plants=plants,
#                                predictions=predictions,
#                                is_logged_in=True)
#     except Exception as e:
#         flash('Error loading dashboard data')
#         return render_template('dashboard.html',
#                                user_name=session.get('user_name', ''),
#                                plants=[],
#                                predictions=[],
#                                is_logged_in=True)
#
#
# @app.route('/predict_disease')
# def predict_disease():
#     is_logged_in = 'user_id' in session
#     return render_template('disease.html', is_logged_in=is_logged_in)
#
#
# @app.route('/predict_watering')
# def predict_watering():
#     is_logged_in = 'user_id' in session
#     return render_template('watering.html', is_logged_in=is_logged_in)
#
#
# @app.route('/my_plants')
# def my_plants():
#     if 'user_id' not in session:
#         return redirect(url_for('home'))
#     return render_template('my_plants.html', is_logged_in=True)
#
#
# @app.route('/community')
# def community():
#     is_logged_in = 'user_id' in session
#     return render_template('community.html', is_logged_in=is_logged_in)
#
#
# @app.route('/profile')
# def profile():
#     if 'user_id' not in session:
#         return redirect(url_for('home'))
#     return render_template('profile.html', is_logged_in=True)
#
#
# @app.route('/history')
# def history():
#     if 'user_id' not in session:
#         return redirect(url_for('home'))
#
#     try:
#         history = list(mongo.db.predictions.find({'userId': ObjectId(session['user_id'])}).sort('timestamp', -1))
#         for item in history:
#             item['_id'] = str(item['_id'])
#             item['userId'] = str(item['userId'])
#         return render_template('history.html', history=history, is_logged_in=True)
#     except Exception as e:
#         flash("Error loading history")
#         return render_template('history.html', history=[], is_logged_in=True)
#
#
# @app.route('/settings')
# def settings():
#     if 'user_id' not in session:
#         return redirect(url_for('home'))
#     return render_template('settings.html', is_logged_in=True)
#
#
# if __name__ == '__main__':
#     try:
#         port = int(os.getenv('PORT', 5000))
#         use_debugger = os.getenv('USE_DEBUGGER', 'true').lower() == 'true'
#
#         from werkzeug.serving import run_simple
#
#         run_simple(
#             'localhost',
#             port,
#             app,
#             use_debugger=use_debugger,
#             use_reloader=False,
#             threaded=True
#         )
#     except Exception as e:
#         print(f"Failed to start server: {e}")
#         raise