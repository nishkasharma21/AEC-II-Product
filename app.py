from flask import Flask, request, jsonify

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


if __name__ == '__main__':
    app.run(host='localhost', port=5001, debug=True)