from flask import Flask, request, jsonify, Response, render_template, session, redirect, url_for
import pyaudio
import cv2
import time
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaRecorder
import torch
import torchvision.transforms as T
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
from webcam import detect_objects
from queue import Queue

output_latency = 0
input_latency = 0

app = Flask(__name__)

# Store received data for display
received_data = {}

@app.route('/environmental-data', methods=['POST'])
def receive_data():
    global received_data
    if request.is_json:
        received_data = request.get_json()
        print(f"Received data: {received_data}")
        return jsonify({"message": "Data received successfully"}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400

@app.route('/get_environmental-data', methods=['GET'])
def get_data():
    print(f"Sending data: {received_data}")
    return jsonify(received_data)

# Global variable to store the latest frame
latest_frame = None

# @app.route('/raspi_to_flask_camera', methods=['POST'])
# def video_feed():
#     global latest_frame
#     try:
#         # Read the frame from the request
#         file = request.files['frame'].read()
#         npimg = np.frombuffer(file, np.uint8)
#         frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#         # Update the latest frame
#         latest_frame = frame

#         return Response(status=200)
#     except Exception as e:
#         print(f"Error processing frame: {e}")
#         return Response(status=500)

# @app.route('/raspi_to_flask_camera', methods=['POST'])
# def video_feed():
#     global latest_frame
#     try:
#         # Read the frame from the request
#         file = request.files['frame'].read()
#         npimg = np.frombuffer(file, np.uint8)
#         frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
#         # Run ML model detection (dummy example here)
#         detections = detect_objects(frame)  # Replace with actual detection function

#         # Draw bounding boxes and labels on the frame
#         for detection in detections:
#             x, y, w, h, label = detection
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         latest_frame = frame
#         return Response(status=200)
#     except Exception as e:
#         print(f"Error processing frame: {e}")
#         return Response(status=500)

# def generate():
#     global latest_frame
#     while True:
#         if latest_frame is not None:
#             _, jpeg = cv2.imencode('.jpg', latest_frame)
#             frame = jpeg.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/video_stream')
# def video_stream():
#     return Response(generate(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# Variables to track gun detection state and timestamps
gun_detected = False
gun_detected_start_time = 0
last_notification_time = 0

# Time thresholds (in seconds)
detection_threshold = 3
notification_cooldown = 30

latest_frame = None

notification_queue = Queue()

@app.route('/raspi_to_flask_camera', methods=['POST'])
def video_feed():
    global latest_frame, gun_detected, gun_detected_start_time, last_notification_time
    try:
        # Read the frame from the request
        file = request.files['frame'].read()
        npimg = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Perform object detection
        detections = detect_objects(frame)

        # Track gun detection
        current_time = time.time()
        gun_present = any(label.startswith('gun') for _, _, _, _, label in detections)

        if gun_present:
            if not gun_detected:
                gun_detected = True
                gun_detected_start_time = current_time
            elif current_time - gun_detected_start_time >= detection_threshold:
                if current_time - last_notification_time >= notification_cooldown:
                    send_notification()
                    last_notification_time = current_time
        else:
            gun_detected = False

        # Draw bounding boxes and labels on the frame
        for detection in detections:
            x, y, w, h, label = detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        latest_frame = frame
        return Response(status=200)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return Response(status=500)

def send_notification():
    # Function to send notification when a gun is detected
    print("Gun detected! Sending notification...")
    notification_data = {
        'message': 'Gun detected!'
    }
    notification_queue.put(notification_data)

def generate():
    global latest_frame
    while True:
        if latest_frame is not None:
            _, jpeg = cv2.imencode('.jpg', latest_frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_stream')
def video_stream():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/notifications', methods=['GET'])
def get_notifications():
    if not notification_queue.empty():
        notification = notification_queue.get()
        return jsonify(notification)
    else:
        return jsonify({'message': 'No new notifications'}), 204

@app.route('/')
def index():
    return render_template('index.html')

# audio1 = pyaudio.PyAudio()

# FORMAT = audio1.get_format_from_width(width=2)
# CHANNELS = 1
# RATE = 44100
# CHUNK = 2048
# RECORD_SECONDS = 5

# def genHeader(sampleRate, bitsPerSample, channels):
#     datasize = 2000*10**6
#     o = bytes("RIFF",'ascii')                                               # (4byte) Marks file as RIFF
#     o += (datasize + 36).to_bytes(4,'little')                               # (4byte) File size in bytes excluding this and RIFF marker
#     o += bytes("WAVE",'ascii')                                              # (4byte) File type
#     o += bytes("fmt ",'ascii')                                              # (4byte) Format Chunk Marker
#     o += (16).to_bytes(4,'little')                                          # (4byte) Length of above format data
#     o += (1).to_bytes(2,'little')                                           # (2byte) Format type (1 - PCM)
#     o += (channels).to_bytes(2,'little')                                    # (2byte)
#     o += (sampleRate).to_bytes(4,'little')                                  # (4byte)
#     o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4,'little')  # (4byte)
#     o += (channels * bitsPerSample // 8).to_bytes(2,'little')               # (2byte)
#     o += (bitsPerSample).to_bytes(2,'little')                               # (2byte)
#     o += bytes("data",'ascii')                                              # (4byte) Data Chunk Marker
#     o += (datasize).to_bytes(4,'little')                                    # (4byte) Data size in bytes
#     return o

# @app.route('/audio')
# def audio():
#     # start Recording
#     def sound():

#         CHUNK = 100
#         sampleRate = 44100
#         bitsPerSample = 16
#         channels = 1
#         wav_header = genHeader(sampleRate, bitsPerSample, channels)
#         stream = audio1.open(format=FORMAT, channels=CHANNELS,
#                         rate=RATE, input=True, output= True, 
#                         frames_per_buffer=CHUNK)
#         print("recording...")
#         #frames = []
#         first_run = True
#         while True:
#            if first_run:
#                data = wav_header + stream.read(CHUNK, exception_on_overflow = False)
#                first_run = False
#            else:
#                data = stream.read(CHUNK, exception_on_overflow = False)
#            yield(data)

#     return Response(sound())

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import secrets

CORS(app)  # Allow CORS for development purposes

app.secret_key = "6c043097f3219be58949aec5856cbbc6"  # Required to use sessions

# Initialize SQLite database
def init_db():
    with sqlite3.connect('users.db') as conn:
        cursor = conn.cursor()
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        
        # Groups table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                admin_id INTEGER NOT NULL UNIQUE,  -- Unique constraint on admin_id
                user_ids TEXT,
                FOREIGN KEY (admin_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()

init_db()

@app.route('/signup', methods=['POST'])
def sign_up():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Email and password are required'}), 400

    try:
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (email, password)
                VALUES (?, ?)
            ''', (email, password))
            conn.commit()
        return jsonify({'message': 'Sign up successful!'}), 200
    except sqlite3.IntegrityError:
        return jsonify({'message': 'Email already exists'}), 400

@app.route('/login', methods=['POST'])
def log_in():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Email and password are required'}), 400

    with sqlite3.connect('users.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM users WHERE email = ? AND password = ?
        ''', (email, password))
        user = cursor.fetchone()

    if user:
        session['user_id'] = user[0]  # Store user id in session
        return jsonify({'message': 'Login successful!', 'user_id': user[0]}), 200
    else:
        return jsonify({'message': 'Invalid email or password'}), 401

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user_id', None)  # Remove the user_id from the session
    return jsonify({'message': 'Logged out successfully!'}), 200

@app.route('/create_group', methods=['POST'])
def create_group():
    if 'user_id' not in session:
        return jsonify({'message': 'Unauthorized'}), 401

    data = request.json
    group_name = data.get('group_name')

    if not group_name:
        return jsonify({'message': 'Group name is required'}), 400

    try:
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO groups (name, admin_id, user_ids)
                VALUES (?, ?, ?)
            ''', (group_name, session['user_id'], str(session['user_id'])))
            group_id = cursor.lastrowid  # Get the id of the newly created group
            conn.commit()
        return jsonify({'message': 'Group created successfully!', 'group_id': group_id}), 200
    except sqlite3.IntegrityError:
        return jsonify({'message': 'Group name already exists'}), 400
    
# @app.route('/groups', methods=['GET'])
# def get_groups():
#     # Ensure user_id is provided in the query parameters
#     user_id = request.args.get('user_id', type=int)
#     if not user_id:
#         return jsonify({"message": "User ID is required"}), 400

#     # Establish a connection to the database
#     conn = sqlite3.connect('users.db')
#     conn.row_factory = sqlite3.Row
#     cursor = conn.cursor()

#     # Fetch the groups for the given user_id
#     cursor.execute('''
#         SELECT u.id, u.email
#         FROM users u
#         JOIN user_groups ug ON u.id = ug.user_id
#         WHERE ug.group_id = (SELECT group_id FROM user_groups WHERE user_id = ? LIMIT 1)
#     ''', (user_id,))
#     users = cursor.fetchall()
#     conn.close()

#     # Format the response
#     result = []
#     for user in users:
#         result.append({
#             "id": user["id"],
#             "email": user["email"]
#         })

#     return jsonify(result)

# @app.route('/add_user_to_group', methods=['POST'])
# def add_user_to_group():
#     if 'user_id' not in session:
#         return jsonify({'message': 'Unauthorized'}), 401

#     data = request.json
#     group_id = data.get('group_id')
#     user_email = data.get('user_email')

#     if not group_id or not user_email:
#         return jsonify({'message': 'Group ID and user email are required'}), 400

#     with sqlite3.connect('users.db') as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#             SELECT admin_id FROM groups WHERE id = ?
#         ''', (group_id,))
#         group = cursor.fetchone()

#         if not group or group[0] != session['user_id']:
#             return jsonify({'message': 'Unauthorized or group not found'}), 401

#         cursor.execute('''
#             SELECT id FROM users WHERE email = ?
#         ''', (user_email,))
#         user = cursor.fetchone()

#         if not user:
#             return jsonify({'message': 'User not found'}), 404

#         try:
#             cursor.execute('''
#                 INSERT INTO user_groups (user_id, group_id)
#                 VALUES (?, ?)
#             ''', (user[0], group_id))
#             conn.commit()
#             return jsonify({'message': 'User added to group successfully!'}), 200
#         except sqlite3.IntegrityError:
#             return jsonify({'message': 'User already in group'}), 400

# @app.route('/remove_user_from_group', methods=['POST'])
# def remove_user_from_group():
#     if 'user_id' not in session:
#         return jsonify({'message': 'Unauthorized'}), 401

#     data = request.json
#     group_id = data.get('group_id')
#     user_email = data.get('user_email')

#     if not group_id or not user_email:
#         return jsonify({'message': 'Group ID and user email are required'}), 400

#     with sqlite3.connect('users.db') as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#             SELECT admin_id FROM groups WHERE id = ?
#         ''', (group_id,))
#         group = cursor.fetchone()

#         if not group or group[0] != session['user_id']:
#             return jsonify({'message': 'Unauthorized or group not found'}), 401

#         cursor.execute('''
#             SELECT id FROM users WHERE email = ?
#         ''', (user_email,))
#         user = cursor.fetchone()

#         if not user:
#             return jsonify({'message': 'User not found'}), 404

#         cursor.execute('''
#             DELETE FROM user_groups WHERE user_id = ? AND group_id = ?
#         ''', (user[0], group_id))
#         conn.commit()
#         return jsonify({'message': 'User removed from group successfully!'}), 200


if __name__ == '__main__':
    app.run(host='172.20.10.2', debug=True, port=8000)

