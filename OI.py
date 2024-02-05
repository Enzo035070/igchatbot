import speech_recognition as sr
audio_file = sr.AudioFile('/Users/enzo/projects/EnzoCodingChatbot/Sector.aiff')
recognizer = sr.Recognizer()
with audio_file as source:
    audio = recognizer.record(source)
text = recognizer.recognize_google(audio)
print(text)