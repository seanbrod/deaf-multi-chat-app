import os
import json
import random  # ✅ Import random for generating numbers
from flask import Flask, send_file, jsonify, request

app = Flask(__name__)

# Load config.json (if needed)
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

index_path = os.path.join(os.path.dirname(__file__), config['paths']['index'])

# Variable to track listening state
listening = False

# Function to generate a random number
def my_python_function():
    return f"Random Number: {random.randint(1, 100)}"  # ✅ Generates random numbers

@app.route('/toggle_listening', methods=['POST'])
def toggle_listening():
    global listening
    listening = not listening
    return jsonify(listening=listening)

@app.route('/get_live_input', methods=['GET'])
def get_live_input():
    global listening
    if listening:
        return jsonify(input=my_python_function())  # ✅ Fetches random numbers
    else:
        return jsonify(input="Listening is OFF")

@app.route('/')
def home():
    return send_file(index_path)

if __name__ == '__main__':
    app.run(debug=True)
