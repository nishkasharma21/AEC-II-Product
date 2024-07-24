from flask import Flask, request, jsonify, Response, render_template
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

output_latency = 0
input_latency = 0

app = Flask(__name__)

# # Store received data for display
# received_data = {}

# # Load your pre-trained PyTorch model
# model = torch.load('/Users/nishkasharma/Desktop/AEC-II-Product/15_epoch.pt')
# model.eval()

# # Define the transformation to preprocess the frames
# transform = T.Compose([
#     T.ToTensor(),
#     T.Resize((640, 640)),  # Resize to the input size expected by your model
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

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

@app.route('/raspi_to_flask_camera', methods=['POST'])
def video_feed():
    global latest_frame
    try:
        # Read the frame from the request
        file = request.files['frame'].read()
        npimg = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Run ML model detection (dummy example here)
        detections = detect_objects(frame)  # Replace with actual detection function

        # Draw bounding boxes and labels on the frame
        for detection in detections:
            x, y, w, h, label = detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        latest_frame = frame
        return Response(status=200)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return Response(status=500)

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
    

# def generate_frames():
#     camera = cv2.VideoCapture(0)  # 0 for the default camera

#     time.sleep(output_latency+input_latency)

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             # Convert frame to PIL Image
#             pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
#             # # Apply transformations
#             # input_tensor = transform(pil_image).unsqueeze(0)

#             # # Run inference
#             # with torch.no_grad():
#             #     outputs = model(input_tensor)
            
#             # Process the outputs
#             # Assuming the model returns a list of dictionaries as output, each containing 'boxes' and 'labels'
#             # for output in outputs:
#             #     boxes = output['boxes'].cpu().numpy().astype(np.int32)
#             #     labels = output['labels'].cpu().numpy()
                
#             #     # Annotate the frame
#             #     for box, label in zip(boxes, labels):
#             #         if label == 1:  # Assuming label 1 is for guns
#             #             cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_video():
    # Replace '0' with your video source, it could be a webcam or a video file
    video_source = 0
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Concatenate frame to be used in streaming response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

@app.route('/')
def index():
    return render_template('index.html')

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import sqlite3

# app = Flask(__name__)
# CORS(app)  # Allow CORS for development purposes

# # Initialize SQLite database
# def init_db():
#     with sqlite3.connect('users.db') as conn:
#         cursor = conn.cursor()
#         # Users table
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS users (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 email TEXT UNIQUE NOT NULL,
#                 password TEXT NOT NULL
#             )
#         ''')
        
#         # Groups table
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS groups (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 name TEXT UNIQUE NOT NULL,
#                 admin_id INTEGER NOT NULL,
#                 FOREIGN KEY (admin_id) REFERENCES users (id)
#             )
#         ''')
        
#         # UserGroups table
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS user_groups (
#                 user_id INTEGER NOT NULL,
#                 group_id INTEGER NOT NULL,
#                 FOREIGN KEY (user_id) REFERENCES users (id),
#                 FOREIGN KEY (group_id) REFERENCES groups (id),
#                 PRIMARY KEY (user_id, group_id)
#             )
#         ''')
        
#         conn.commit()

# init_db()

# @app.route('/signup', methods=['POST'])
# def sign_up():
#     data = request.json
#     email = data.get('email')
#     password = data.get('password')

#     if not email or not password:
#         return jsonify({'message': 'Email and password are required'}), 400

#     try:
#         with sqlite3.connect('users.db') as conn:
#             cursor = conn.cursor()
#             cursor.execute('''
#                 INSERT INTO users (email, password)
#                 VALUES (?, ?)
#             ''', (email, password))
#             conn.commit()
#         return jsonify({'message': 'Sign up successful!'}), 200
#     except sqlite3.IntegrityError:
#         return jsonify({'message': 'Email already exists'}), 400

# @app.route('/login', methods=['POST'])
# def log_in():
#     data = request.json
#     email = data.get('email')
#     password = data.get('password')

#     if not email or not password:
#         return jsonify({'message': 'Email and password are required'}), 400

#     with sqlite3.connect('users.db') as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#             SELECT * FROM users WHERE email = ? AND password = ?
#         ''', (email, password))
#         user = cursor.fetchone()

#     if user:
#         return jsonify({'message': 'Login successful!', 'user_id': user[0]}), 200
#     else:
#         return jsonify({'message': 'Invalid email or password'}), 401

# @app.route('/create_group', methods=['POST'])
# def create_group():
#     data = request.json
#     user_id = data.get('user_id')  # Admin user
#     group_name = data.get('name')

#     if not user_id or not group_name:
#         return jsonify({'message': 'User ID and group name are required'}), 400

#     with sqlite3.connect('users.db') as conn:
#         cursor = conn.cursor()
#         try:
#             cursor.execute('''
#                 INSERT INTO groups (name, admin_id)
#                 VALUES (?, ?)
#             ''', (group_name, user_id))
#             group_id = cursor.lastrowid
#             cursor.execute('''
#                 INSERT INTO user_groups (user_id, group_id)
#                 VALUES (?, ?)
#             ''', (user_id, group_id))
#             conn.commit()
#             return jsonify({'message': 'Group created successfully!', 'group_id': group_id}), 200
#         except sqlite3.IntegrityError:
#             return jsonify({'message': 'Group name already exists'}), 400
        
# @app.route('/users', methods=['GET'])
# def get_users():
#     with sqlite3.connect('users.db') as conn:
#         cursor = conn.cursor()
#         cursor.execute('SELECT id, email FROM users')
#         users = cursor.fetchall()
#     return jsonify([{'id': user[0], 'email': user[1]} for user in users])

# @app.route('/groups/<int:group_id>/members', methods=['GET'])
# def get_group_members(group_id):
#     with sqlite3.connect('users.db') as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#             SELECT u.id, u.email FROM users u
#             JOIN user_groups ug ON u.id = ug.user_id
#             WHERE ug.group_id = ?
#         ''', (group_id,))
#         members = cursor.fetchall()
#     return jsonify([{'id': member[0], 'email': member[1]} for member in members])

# @app.route('/add_user_to_group', methods=['POST'])
# def add_user_to_group():
#     data = request.json
#     admin_id = data.get('admin_id')
#     user_id = data.get('user_id')
#     group_id = data.get('group_id')

#     if not admin_id or not user_id or not group_id:
#         return jsonify({'message': 'Admin ID, user ID, and group ID are required'}), 400

#     with sqlite3.connect('users.db') as conn:
#         cursor = conn.cursor()
        
#         # Check if the group exists and if the admin is the correct one
#         cursor.execute('SELECT admin_id FROM groups WHERE id = ?', (group_id,))
#         group = cursor.fetchone()
#         if not group:
#             return jsonify({'message': 'Group does not exist'}), 404
        
#         if group[0] != admin_id:
#             return jsonify({'message': 'Only the group admin can add users'}), 403
        
#         # Check if the user is already in the group
#         cursor.execute('''
#             SELECT * FROM user_groups WHERE user_id = ? AND group_id = ?
#         ''', (user_id, group_id))
#         existing_membership = cursor.fetchone()
#         if existing_membership:
#             return jsonify({'message': 'User is already in this group'}), 400

#         # Add the user to the group
#         try:
#             cursor.execute('''
#                 INSERT INTO user_groups (user_id, group_id)
#                 VALUES (?, ?)
#             ''', (user_id, group_id))
#             conn.commit()
#             return jsonify({'message': 'User added to group successfully!'}), 200
#         except sqlite3.IntegrityError as e:
#             return jsonify({'message': f'Error adding user to group: {str(e)}'}), 400


# @app.route('/remove_user_from_group', methods=['POST'])
# def remove_user_from_group():
#     data = request.json
#     admin_id = data.get('admin_id')
#     user_id = data.get('user_id')
#     group_id = data.get('group_id')

#     if not admin_id or not user_id or not group_id:
#         return jsonify({'message': 'Admin ID, user ID, and group ID are required'}), 400

#     with sqlite3.connect('users.db') as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#             SELECT admin_id FROM groups WHERE id = ?
#         ''', (group_id,))
#         group = cursor.fetchone()
#         if not group or group[0] != admin_id:
#             return jsonify({'message': 'Only group admin can remove users'}), 403

#         cursor.execute('''
#             DELETE FROM user_groups
#             WHERE user_id = ? AND group_id = ?
#         ''', (user_id, group_id))
#         conn.commit()
        
#         if cursor.rowcount > 0:
#             return jsonify({'message': 'User removed from group successfully!'}), 200
#         else:
#             return jsonify({'message': 'User is not in this group'}), 404

if __name__ == '__main__':
    app.run(host='172.20.10.2', debug=True, port=8000)

