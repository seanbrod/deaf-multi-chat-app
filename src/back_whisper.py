#backend using onnxruntime and openAI whisper quantized
import os, json
import onnxruntime as ort
from onnxruntime_extensions import get_library_path
import numpy as np

#DOCS
#https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html Onnxruntime CoreML


with open('config.json', 'r') as file:
    config = json.load(file)

model_path = os.path.join(config['base']['assets'], config['paths']['model'])
data_test_path = os.path.join(config['base']['data'], config['paths']['testF'])


#this function will take in a .wav file and transcibe it using whisper model. 
def transcribe(): 

    provider = [
        ('CoreMLExecutionProvider', {
            "ModelFormat": "MLProgram", "MLComputeUnits": "ALL", 
            "RequireStaticInputShapes": "0", "EnableOnSubgraphs": "0"
        }),
    ]

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
    session = ort.InferenceSession(model_path, options, providers=provider)
    inputs.pop("attention_mask", None)
    outputs = session.run(None, inputs)[0]

def main():
    transcribe()

if __name__ == "__main__":
    main()

