import speech_recognition as sr

def record_and_recognize():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Dinleniyor...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language="tr-TR")
        return text
    except sr.UnknownValueError:
        print("Ses anlaşılamadı.")
        return ""
    except sr.RequestError:
        print("API hatası")
        return""