import os
from flask import Flask, render_template, redirect, url_for, flash, request
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField
from wtforms.validators import DataRequired, EqualTo
from werkzeug.utils import secure_filename
from datetime import datetime
import joblib
import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf



# Initialize Flask app
app = Flask(__name__)

# Secret key for session management
app.config['SECRET_KEY'] = 'your_secret_key_here'

# SQLite database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Path to save the audio files
AUDIO_SAVE_PATH = os.path.join(os.getcwd(), 'audio_files')
if not os.path.exists(AUDIO_SAVE_PATH):
    os.makedirs(AUDIO_SAVE_PATH)

# Path to load the trained model
MODEL_PATH = r"E:\Python program\infant_pain_classifier.pkl"  # Update to your actual trained model path

# Load the trained model
model = joblib.load(MODEL_PATH)

# Database model for storing user information
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    infant_name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)

# Form for Registration
class RegistrationForm(FlaskForm):
    infant_name = StringField('Infant Name', validators=[DataRequired()])
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Role', choices=[('Parent', 'Parent'), ('Caregiver', 'Caregiver'), ('Healthcare Professional', 'Healthcare Professional')], validators=[DataRequired()])

# Form for Login
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])

# Home route
@app.route('/')
def home():
    return redirect(url_for('login'))

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('Username already exists, please choose another one.', 'danger')
            return render_template('register.html', form=form)

        new_user = User(infant_name=form.infant_name.data, username=form.username.data, password=form.password.data, role=form.role.data)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data, password=form.password.data).first()
        if user:
            flash('Login successful!', 'success')
            return redirect(url_for('record_page'))  # Redirect to the recording page
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html', form=form)

# Route to display the recording page
@app.route('/record_page')
def record_page():
    return render_template('record.html')

# Extract MFCC features from the audio file
# Extract MFCC features from the audio file using librosa
def extract_features(file_name):
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_name, sr=None, res_type='kaiser_fast')
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

# Function to load audio using pydub and convert to numpy array
def load_audio_pydub(file_path):
    # Load the audio using pydub
    audio = AudioSegment.from_wav(file_path)
    # Convert to numpy array for librosa processing
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    return samples, audio.frame_rate

# Record audio route

# Record audio route
@app.route('/record', methods=['POST'])
def record():
    if 'audio_data' not in request.files:
        flash('No audio data received', 'danger')
        return redirect(url_for('record_page'))

    # Get the uploaded audio file
    audio_file = request.files['audio_data']

    # Secure the file name and save it temporarily
    temp_path = os.path.join(AUDIO_SAVE_PATH, 'temp_audio')
    audio_file.save(temp_path)

    # Load and convert the file to .wav using pydub (regardless of its original format)
    try:
        audio = AudioSegment.from_file(temp_path)
        audio_save_path = os.path.join(AUDIO_SAVE_PATH, 'child_cry.wav')  # Save as child_cry.wav
        audio.export(audio_save_path, format='wav')
    except Exception as e:
        flash(f"Error processing audio file: {e}", 'danger')
        return redirect(url_for('record_page'))

    # Clean up temporary file
    os.remove(temp_path)

    flash('Recording saved successfully as .wav!', 'success')
    return redirect(url_for('result', audio_file='child_cry.wav'))

# Function to load audio using soundfile
def load_audio_soundfile(file_path):
    audio, sample_rate = sf.read(file_path)
    return audio, sample_rate


# Update your result function to use the new audio loading method
@app.route('/result')
def result():
    audio_file = request.args.get('audio_file')
    if not audio_file:
        flash('No audio file found.', 'danger')
        return redirect(url_for('record_page'))

    # Full path to the saved audio file
    audio_path = os.path.join(AUDIO_SAVE_PATH, audio_file)

    # Load audio using soundfile
    print(f"Loading audio file from: {audio_path}")
    audio, sample_rate = load_audio_soundfile(audio_path)

    # Extract features from the audio
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)

    # Make a prediction using the trained model
    prediction = model.predict([mfccs_scaled])

    # Interpret the prediction
    if prediction[0] == 3:
        result_text = "The pain is caused by belly pain."
    elif prediction[0] == 4:
        result_text = "The pain is caused by burping."
    elif prediction[0] == 0:
        result_text = "The pain is caused by discomfort."
    elif prediction[0] == 2:
        result_text = "The pain is caused by hunger."
    else:
        result_text = "The pain is caused by tiredness."

    # Render the result page with the prediction
    return render_template('result.html', result=result_text)
if __name__ == '__main__':
    app.run(debug=True)

