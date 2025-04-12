import pyttsx3

class TextToSpeech:
    def __init__(self, rate=150, volume=1.0, voice_idx=0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[voice_idx].id)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()