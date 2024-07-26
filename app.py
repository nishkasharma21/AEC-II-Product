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
import json

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

from flask import Flask, request, jsonify, session
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
                event_ids TEXT, 
                FOREIGN KEY (admin_id) REFERENCES users (id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                date TEXT NOT NULL,  -- Using TEXT to store date in ISO 8601 format (e.g., "2024-07-25")
                video TEXT           -- Store video URL or path
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

            # Get the user_id of the newly inserted user
            cursor.execute('SELECT last_insert_rowid()')
            user_id = cursor.fetchone()[0]

            # Store the user_id in the session
            session['user_id'] = user_id

        return jsonify({'message': 'Sign up successful!', 'user_id': user_id}), 200
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
        return jsonify({'message': 'Unauthorized', 'status': 401}), 401

    data = request.json
    group_name = data.get('group_name')

    if not group_name:
        return jsonify({'message': 'Group name is required', 'status': 400}), 400

    try:
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            # Check if the user has already created a group
            cursor.execute('''
                SELECT id FROM groups WHERE admin_id = ?
            ''', (session['user_id'],))
            existing_group = cursor.fetchone()

            if existing_group:
                return jsonify({'message': 'User already created a group', 'status': 400}), 400

            cursor.execute('''
                INSERT INTO groups (name, admin_id)
                VALUES (?, ?)
            ''', (group_name, session['user_id']))
            group_id = cursor.lastrowid  # Get the id of the newly created group
            print(group_id)
            conn.commit()
        return jsonify({'message': 'Group created successfully!', 'group_id': group_id, 'status': 200}), 200
    except sqlite3.IntegrityError as e:
        # Log the specific error for debugging purposes
        print(f'SQLite Integrity Error: {e}')
        return jsonify({'message': 'Group name already exists', 'status': 400}), 400
    except Exception as e:
        # Log any other exceptions that occur
        print(f'Unexpected Error: {e}')
        return jsonify({'message': 'Internal server error', 'status': 500}), 500

@app.route('/add_user_to_group', methods=['POST'])
def add_user_to_group():
    if 'user_id' not in session:
        return jsonify({'message': 'Unauthorized'}), 401

    data = request.json
    group_id = data.get('group_id')
    user_email = data.get('user_email')

    if not group_id or not user_email:
        return jsonify({'message': 'Group ID and user email are required'}), 400

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    try:
        # Get the admin ID for the group
        cursor.execute('''
            SELECT admin_id FROM groups WHERE id = ?
        ''', (group_id,))
        group = cursor.fetchone()

        if not group:
            return jsonify({'message': 'Group not found'}), 404
        
        if group[0] != session['user_id']:
            return jsonify({'message': 'Unauthorized'}), 401

        # Get the user ID for the provided email
        cursor.execute('''
            SELECT id FROM users WHERE email = ?
        ''', (user_email,))
        user = cursor.fetchone()

        if not user:
            return jsonify({'message': 'User not found'}), 404

        user_id = user[0]

        # Get the current user ID
        current_user_id = session['user_id']

        # Get the list of user IDs already in the group
        cursor.execute('''
            SELECT user_ids FROM groups WHERE id = ?
        ''', (group_id,))
        group = cursor.fetchone()

        user_ids = group[0]
        if user_ids:
            user_ids_list = user_ids.split(',')
        else:
            user_ids_list = []

        print(user_ids_list)

        if str(user_id) not in user_ids_list:
            user_ids_list.append(str(user_id))
            updated_user_ids = ','.join(user_ids_list)

            cursor.execute('''
                UPDATE groups SET user_ids = ? WHERE id = ?
            ''', (updated_user_ids, group_id))
            conn.commit()

            return jsonify({'message': 'User added to group successfully!'}), 200
        else:
            return jsonify({'message': 'User already in group'}), 400

    except sqlite3.Error as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500

    finally:
        conn.close()


@app.route('/get_group_emails', methods=['GET'])
def get_group_emails():
    group_id = request.args.get('group_id')

    if not group_id:
        return jsonify({'message': 'Group ID is required'}), 400

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT user_ids FROM groups WHERE id = ?
        ''', (group_id,))
        group = cursor.fetchone()

        if not group or not group[0]:
            return jsonify({'message': 'Group not found or no users in group'}), 404

        user_ids = group[0].split(',')
        if not user_ids:
            return jsonify({'emails': []}), 200

        cursor.execute(f'''
            SELECT email FROM users WHERE id IN ({','.join(['?']*len(user_ids))})
        ''', user_ids)
        emails = cursor.fetchall()

        email_list = [email[0] for email in emails]

        return jsonify({'emails': email_list}), 200

    except sqlite3.Error as e:
        return jsonify({'error': str(e)}), 500

    finally:
        conn.close()


@app.route('/remove_user_from_group', methods=['POST'])
def remove_user_from_group():
    data = request.get_json()
    group_id = data.get('group_id')
    email = data.get('email')

    if not group_id or not email:
        return jsonify({'message': 'Group ID and Email are required'}), 400

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    try:
        # Get the user ID from the email
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        if not user:
            return jsonify({'message': 'User not found'}), 404
        user_id = str(user[0])

        # Get the group admin ID and current user IDs
        cursor.execute('''
            SELECT admin_id, user_ids FROM groups WHERE id = ?
        ''', (group_id,))
        group = cursor.fetchone()
        if not group:
            return jsonify({'message': 'Group not found'}), 404

        group_admin_id, user_ids = group
        print(group_admin_id)
        print(user_id)

        current_user_id = session['user_id']
        if str(current_user_id) != str(group_admin_id):
            return jsonify({'message': 'Only the group admin can remove users'}), 403

        # Get the user ID of the user to be removed
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        user_to_remove = cursor.fetchone()
        if not user_to_remove:
            return jsonify({'message': 'User not found'}), 404
        user_to_remove_id = str(user_to_remove[0])

        # Remove the user ID from the list
        user_ids_list = user_ids.split(',')
        if user_to_remove_id in user_ids_list:
            user_ids_list.remove(user_to_remove_id)
        else:
            return jsonify({'message': 'User not in group'}), 404

        # Update the group with the new user_ids
        new_user_ids = ','.join(user_ids_list)
        cursor.execute('UPDATE groups SET user_ids = ? WHERE id = ?', (new_user_ids, group_id))
        conn.commit()

        return jsonify({'message': 'User removed from group'}), 200

    except sqlite3.Error as e:
        return jsonify({'error': str(e)}), 500

    finally:
        conn.close()

@app.route('/get_user_groups', methods=['GET'])
def get_user_groups():
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({'message': 'User ID is required'}), 400

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    try:
        # Fetch groups where the user is an admin
        cursor.execute('''
            SELECT id, name FROM groups WHERE admin_id = ?
        ''', (user_id,))
        groups = cursor.fetchall()

        if not groups:
            return jsonify({'admin_groups': []}), 200

        # Create a list of dictionaries with groupId and groupName
        admin_groups = [{'id': group[0], 'name': group[1]} for group in groups]

        return jsonify({'admin_groups': admin_groups}), 200

    except sqlite3.Error as e:
        return jsonify({'error': str(e)}), 500

    finally:
        conn.close()

@app.route('/get_group_events', methods=['GET'])
def get_group_events():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'message': 'User ID is required'}), 400

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    try:
        # Fetch the group events
        cursor.execute('''
            SELECT id, event_ids FROM groups WHERE admin_id = ? OR user_ids LIKE ?
        ''', (str(user_id), f'%,{user_id},%'))
        groups = cursor.fetchall()

        if not groups:
            return jsonify({'message': 'No groups found for the user'}), 404

        # Collect events for each group
        events = []
        for group in groups:
            group_id, event_ids = group
            if event_ids:
                event_ids_list = event_ids.split(',')
            else:
                event_ids_list = []

            # Fetch events by IDs
            for event_id in event_ids_list:
                cursor.execute('SELECT * FROM events WHERE id = ?', (event_id,))
                event = cursor.fetchone()
                if event:
                    events.append({
                        'id': event[0],
                        'type': event[1],
                        'date': event[2],
                        'video': event[3]
                    })

        conn.close()
        return jsonify({'events': events}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/create_event', methods=['POST'])
def create_event():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'message': 'No JSON data received'}), 400

        event_type = data.get('type')
        event_date = data.get('date')
        event_video = data.get('video')
        #user_id = session.get('user_id')
        user_id = 1  # Use the appropriate user ID from your authentication logic

        if not event_type or not event_date or not event_video:
            return jsonify({'message': 'All event details are required'}), 400

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Insert the new event into the events table
        cursor.execute('''
            INSERT INTO events (type, date, video)
            VALUES (?, ?, ?)
        ''', (event_type, event_date, event_video))
        event_id = cursor.lastrowid

        # Fetch the current event IDs for the user's groups
        cursor.execute('''
            SELECT id, event_ids FROM groups WHERE admin_id = ? OR user_ids LIKE ?
        ''', (user_id, f'%,{user_id},%'))
        groups = cursor.fetchall()

        if not groups:
            return jsonify({'message': 'No groups found for the user'}), 404

        # Update event IDs for each group
        for group in groups:
            group_id, event_ids = group
            if event_ids:
                event_ids_list = event_ids.split(',')
            else:
                event_ids_list = []
            event_ids_list.append(str(event_id))
            new_event_ids = ','.join(event_ids_list)
            cursor.execute('UPDATE groups SET event_ids = ? WHERE id = ?', (new_event_ids, group_id))

        conn.commit()
        conn.close()

        return jsonify({'message': 'Event created successfully', 'event_id': event_id}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_user_group_status', methods=['GET'])
def check_user_group_status():
    try:
        user_id = session.get('user_id')  # Get the current user's ID from the authentication system

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Check if the user is an admin of any group
        cursor.execute('''
            SELECT id FROM groups WHERE admin_id = ?
        ''', (user_id,))
        is_admin = cursor.fetchone() is not None

        # Check if the user is a member of any group
        cursor.execute('''
            SELECT group_id FROM group_members WHERE user_id = ?
        ''', (user_id,))
        is_member = cursor.fetchone() is not None

        conn.close()

        # Return JSON response indicating if the user is either an admin or a member
        return jsonify({'status': is_admin or is_member})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_user_group', methods=['GET'])
def get_user_group():
    try:
        # For demonstration, assume the user ID is provided in the query parameters
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'message': 'User ID is required'}), 400

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Query to find the group where the user is either an admin or a member
        cursor.execute('''
            SELECT g.name
            FROM groups g
            WHERE g.admin_id = ? OR g.user_ids LIKE ?
        ''', (user_id, f'%,{user_id},%'))

        group = cursor.fetchone()
        conn.close()

        if group:
            return jsonify({'group_name': group[0]}), 200
        else:
            return jsonify({'message': 'User is not part of any group'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='192.168.1.95', debug=True, port=8000)

