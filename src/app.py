import os, json, logging, threading 
from flask import Flask, send_file, jsonify, request
from back_whisper import run
from value_singleton import shared_value

app = Flask(__name__)

# Load config.json (if needed)
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

log_path = config['paths']['log']
# LOGGING
with open(log_path, 'w'):
    pass
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

index_path = os.path.join(os.path.dirname(__file__), config['paths']['index'])

# Variable to track listening state
listening = False

def run_in_background():
    run('foo')  # This will run in a separate thread

@app.route('/toggle_listening', methods=['POST'])
def toggle_listening():
    logger.info("Toggle Start")
    global listening
    listening = not listening
    if listening:
        logger.info("Thread Starting")
        thread = threading.Thread(target=run_in_background)
        thread.start()
        logger.info("Thread Running")
    else:
        run('exit')
    return jsonify(listening=listening)

@app.route('/get_live_input', methods=['GET'])
def get_live_input():
    logger.info("Live Input Start")
    global listening
    if listening:
        return jsonify(input=shared_value.get_value()) 
    else:
        return jsonify(input="Listening is OFF")

@app.route('/')
def home():
    return send_file(index_path)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    logger.info("Starting App")
