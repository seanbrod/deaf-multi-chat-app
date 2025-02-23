#backend using onnxruntime and openAI whisper quantized
import os, json, logging, wave, threading
import onnxruntime as ort
from onnxruntime_extensions import get_library_path
import numpy as np
import sounddevice as sd
from value_singleton import shared_value

#DOCS


with open('config.json', 'r') as file:
    config = json.load(file)

log_path = config['paths']['log']
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_path, level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

model_path = os.path.join(config['base']['ed'], config['base']['assets'], config['paths']['model_cpu'])
data_test_path = os.path.join(config['base']['ed'], config['base']['data'], config['paths']['testF'])
data_path = os.path.join(config['base']['ed'], config['base']['data'])
duration = config["constants"]["duration"]
sample_rate = config["constants"]["smp_rate"]
channel = config["constants"]["channels"]
device_id = config["constants"]["dev_id"]
running = config["constants"]["run"]
#this is a global varible for holding the names of the wav files
recorded_files = []


#this function will take in a .wav file and transcibe it using whisper model. 
def transcribe(file): 

    with open(file, "rb") as f:
        audio = np.asarray(list(f.read()), dtype=np.uint8)

    inputs = {
        "audio_stream": np.array([audio]),
        "max_length": np.array([30], dtype=np.int32),
        "min_length": np.array([1], dtype=np.int32),
        "num_beams": np.array([5], dtype=np.int32),
        "num_return_sequences": np.array([1], dtype=np.int32),
        "length_penalty": np.array([1.0], dtype=np.float32),
        "repetition_penalty": np.array([1.0], dtype=np.float32),
        "attention_mask": np.zeros((1, 80, 3000), dtype=np.int32),
    }

    options = ort.SessionOptions()
    options.register_custom_ops_library(get_library_path())
    #we choose out execution provider here
    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
    inputs.pop("attention_mask", None)
    outputs = session.run(None, inputs)[0]

    #here we have to get the output(ndarray) to a string for pyside6 to print it.
    if isinstance(outputs, np.ndarray):
        text_output = outputs[0].item()  
    else:
        text_output = str(outputs[0])  

    print(text_output)
    shared_value.update_str(text_output)

    try:
        os.remove(file)
        logger.info("File Deleted")
    except Exception as e:
        logger.exception(f"File deletion exception: {e}")

def transcription_worker():
    while running:
        if recorded_files:
            file = recorded_files.pop(0)  
            transcribe(file)

#this function is responsible for the actual creation of the .wav files
def record_audio():
    global running
    file_index = 1

    while running:
        print(f"Recording file {file_index} from microphone...")

        #here is where we handle the input stream. the channels were messing up
        with sd.InputStream(samplerate=sample_rate, channels=channel, dtype='int16',
                            device=device_id, blocksize=1024, latency='low') as stream:
            audio_data = stream.read(int(duration * sample_rate))[0]  

        filename = os.path.join(data_path, f"clip_{file_index}.wav")
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channel)
            wf.setsampwidth(2)  
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        print(f"Saved: {filename}")
        recorded_files.append(filename)  
        file_index += 1
        #keeps running and recording user input audio from pc and saves files

def user_input(input):
    global running
    while running:
        command = input
        if command.lower() == "exit":
            running = False
            print("Stopping recording...")
            logger.info("Transcription Stopped")

def run(cmd):
    global running
    logger.info("Running...")

    recording_thread = threading.Thread(target=record_audio)
    recording_thread.daemon = True
    recording_thread.start()

    transcription_thread = threading.Thread(target=transcription_worker())
    transcription_thread.daemon = True
    transcription_thread.start()

    print("Type 'exit' to stop the recording.")
    user_input(cmd)

    recording_thread.join()
    transcription_thread.join()

    logger.info("Recording has stopped. All files saved and transcribed.")


