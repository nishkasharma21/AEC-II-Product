from flask import Flask, request, jsonify, Response, render_template, session, send_file
import pyaudio
import torch
import os
import cv2
import time
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaRecorder
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from queue import Queue


#Non-testing stuff
video_path = "None"
preview_path = "None"

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
        
        if 'air_pressure' in received_data:
            air_pressure_pascals = received_data['hPa']
            air_pressure_atm = (air_pressure_pascals / 1013250)
            received_data['hPa'] = air_pressure_atm
        
        print(f"Received data: {received_data}")
        return jsonify({"message": "Data received successfully", "data": received_data}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400

@app.route('/get_environmental-data', methods=['GET'])
def get_data():
    print(f"Sending data: {received_data}")
    return jsonify(received_data)


notification_queue = Queue()

# Set model path to point to the correct path for best.pt
model_path = os.path.join('exp7', 'weights', 'best.pt')

# Load the pre-trained YOLOv5 model with the fixed checkpoint
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')

# Set the confidence threshold
model.conf = 0.35

def detect_objects(frame):
    """
    Perform object detection on the provided frame using YOLOv5.
    
    Args:
        frame (numpy.ndarray): The frame to process.

    Returns:
        list: List of detection results. Each result is a tuple (xmin, ymin, xmax, ymax, confidence, label).
    """
    # Perform inference
    results = model(frame)

    detections = []
    # Parse results
    for pred in results.pred[0]:
        xmin, ymin, xmax, ymax, conf, cls = pred
        if conf >= 0.35:  # Apply confidence threshold
            label = results.names[int(cls)]
            detections.append((int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin), f'{label} {conf:.2f}'))
    
    return detections

append_frame = False
gun_detected_start_time = None
gun_detection_end_time = 0
last_notification_time = 0
file_name = 1

# Time thresholds (in seconds)
detection_threshold = 2
notification_cooldown = 60
recording_duration = 15
video_writer = None
latest_frame = None

def save_thumbnail(preview_path, frame):
    global video_writer
    cv2.imwrite(preview_path, frame)
    video_writer.write(frame)

@app.route('/raspi_to_flask_camera', methods=['POST'])
def video_feed():
    global latest_frame, append_frame, gun_detected_start_time, gun_detection_end_time, last_notification_time, video_writer, file_name, video_path, preview_path, notification_cooldown, recording_duration

    try:
        # Read the frame from the request
        file = request.files['frame'].read()
        npimg = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Perform object detection
        detections = detect_objects(frame)

        # Track gun detection
        current_time = time.time()

        # Check for gun detection
        append_frame = current_time <= gun_detection_end_time

        # Process detections
        new_gun_present = any(label.startswith('gun') or label.startswith('rifle') for _, _, _, _, label in detections)
        if(new_gun_present):
            print("THERES A GUN! RUN!!!")
        # no guns past or present - do nothing
        # a gun in the past, not in the present - append
        # a gun now cooldown in effect - do nothing
        # a gun now cooldown elapsed - create a new one

        if append_frame:
            # A gun is in the past and we're still recording our 15sec
            video_writer.write(frame) # append the frame!
            print("Appending the frame!")
        elif video_writer is not None:
            print("Killing the writer")
            # We just finished recording our 15sec, so deconstruct
            # Stop recording after the recording duration
            video_writer.release()
            video_writer = None

        # A gun is in the present, but we are outside of our 15sec recording
        time_since_last_notification = current_time - last_notification_time
        collect_new_recording = new_gun_present and not append_frame
        if collect_new_recording and time_since_last_notification >= notification_cooldown:
            # Create a new recording
            gun_detected_start_time = current_time
            gun_detection_end_time = current_time + recording_duration
            # Initialize video writer and reset the first frame flag
            file_name += 1
            video_path = 'gun_detected_video'+str(file_name)+'.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_path, fourcc, 1, (frame.shape[1], frame.shape[0]))
            preview_path = 'first_frame_'+str(file_name)+'.jpg'
            print(f"Collect a new recording at {video_path}")
            last_notification_time = current_time
            save_thumbnail(preview_path, frame)
            send_notification()


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
    
audio_queue = Queue()

# Audio configuration
FORMAT = 'int16'
CHANNELS = 1
RATE = 44100
CHUNK = 1024

def generate_header(sample_rate, bits_per_sample, channels):
    datasize = 2000 * 10**6
    o = bytes("RIFF", 'ascii')
    o += (datasize + 36).to_bytes(4, 'little')
    o += bytes("WAVE", 'ascii')
    o += bytes("fmt ", 'ascii')
    o += (16).to_bytes(4, 'little')
    o += (1).to_bytes(2, 'little')
    o += (channels).to_bytes(2, 'little')
    o += (sample_rate).to_bytes(4, 'little')
    o += (sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little')
    o += (channels * bits_per_sample // 8).to_bytes(2, 'little')
    o += (bits_per_sample).to_bytes(2, 'little')
    o += bytes("data", 'ascii')
    o += (datasize).to_bytes(4, 'little')
    return o

@app.route('/audio', methods=['POST'])
def audio():
    data = request.data
    audio_queue.put(data)
    return '', 204

@app.route('/audio_feed')
def audio_feed():
    def generate_audio():
        header = generate_header(RATE, 16, CHANNELS)
        yield header
        while True:
            data = audio_queue.get()
            yield data
    return Response(generate_audio(), mimetype='audio/wav')

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
                video TEXT,           -- Store video URL or path
                preview TEXT
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

    
@app.route('/video/<video>')
def video(video):
    return send_file(video, mimetype='video/mp4')

@app.route('/preview/<preview>')
def preview(preview):
    return send_file(preview, mimetype='image/jpeg')

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
                        'video': event[3],
                        'preview': event[4]
                    })

        conn.close()
        return jsonify({'events': events}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/create_event', methods=['POST'])
def create_event():
    global video_path, preview_path
    try:
        data = request.get_json()
        if not data:
            return jsonify({'message': 'No JSON data received'}), 400

        event_type = data.get('type')
        event_date = data.get('date')
        event_video = video_path
        event_preview = preview_path
        print(event_video)
        print(event_preview)
        #user_id = session.get('user_id')
        user_id = 1  # Use the appropriate user ID from your authentication logic

        if not event_type or not event_date or not event_video or not event_preview:
            return jsonify({'message': 'All event details are required'}), 400

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Insert the new event into the events table
        cursor.execute('''
            INSERT INTO events (type, date, video, preview)
            VALUES (?, ?, ?, ?)
        ''', (event_type, event_date, event_video, event_preview))
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

        return jsonify({'message': 'Event created successfully', 'event_id': event_id}), 200

    except Exception as e:
        print(str(e))
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
            SELECT id FROM groups WHERE user_ids LIKE ?
        ''', (f'%,{user_id},%',))
        is_member = cursor.fetchone() is not None

        conn.close()

        status = is_admin or is_member
        # Return JSON response indicating if the user is either an admin or a member
        return jsonify({'status': status})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_user_group', methods=['GET'])
def get_user_group():
    try:
        # Assuming user_id is stored in session for the logged-in user
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

    except sqlite3.Error as e:
        # Handle specific SQLite errors
        return jsonify({'error': f'SQLite error: {str(e)}'}), 500
    except Exception as e:
        # Handle general errors
        return jsonify({'error': f'General error: {str(e)}'}), 500
    
@app.route('/get_user_emails', methods=['GET'])
def get_user_emails():
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Retrieve all user emails
        cursor.execute('SELECT email FROM users')
        emails = [row[0] for row in cursor.fetchall()]

        conn.close()

        # Return JSON response with all user emails
        return jsonify({'emails': emails})

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500


host_ip = '172.20.10.2'

if __name__ == '__main__':
    app.run(host=host_ip, debug=True, port=8000, use_reloader=False)

