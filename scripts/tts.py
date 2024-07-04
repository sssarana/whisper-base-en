import os               # Import for reading files
from gtts import gTTS

def text_to_speech(text, output_file):
    
    language = 'en'
    speech_object = gTTS(text=text, lang=language, slow=False)


    speech_object.save("/home/whisper/data/input_audio/test.mp3")

text = "I want to test whisper"
output_file = "/home/whisper/data/input_audio/test.mp3"
text_to_speech(text, output_file)
