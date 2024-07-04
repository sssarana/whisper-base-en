# Deploying speech to text model on RB5 platform
This project is based on the following model from AI Hub:
https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/whisper_base_en

Connect to the RB5 via ssh or adb and follow these steps:

## Install ai hub models
```sh
pip install "qai_hub_models[whisper_base_en]"
```
If you have troubles installing samplerate it’s likely because you don’t have cmake installed, to fix this:
```sh
apt-get update
apt-get install cmake
```
Another issue you might face is that you don’t have whisper installed, which can be done by running the following:
```sh
pip install openai-whisper
```

## Run the demo app to ensure it performs fine
```sh
python3 -m qai_hub_models.models.whisper_base_en.demo
```
## Export the model for on-device deployment
```sh
python3 -m qai_hub_models.models.whisper_base_en.export
```

## Dowmload weights
Press the link in the end of the output from the previous commandand  find the -download Output dataset- button on the website with your results, this will download a .h5 model weights that we can then use to run on our device. I provide weights generated for my device, but it's worth downloading your own.

## Clone the repository and install requirements
```sh
git clone https://github.com/sssarana/whisper-base-en.git
pip install -r requirements.txt
```

## Convert model to .tflite
Navigate to the directory and run conversion script. You might need to adjust paths before running it.
```sh
cd whisper-base-en/
python3 scripts/convert.py
```
This will generate saved_model.pb and other necessary files

## Generate test audio file
Run the tts script. Adjust paths as needed, you can also modify the text that you want to be converted to speech, see the code for more.
```sh
python3 scripts/tts.py
```
This will generate test.mp3 file with the sample audio.

## Run the model
soon