import speech_recognition as sr
import pyttsx3
import torch
import openai
import sounddevice as sd

openai.api_key = "OPENAI_API_TOKEN"



#  LISTEN

def listen():
    r = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language="ru-RU")

    except Exception as e:
        print(e)
        return "---"    
    return query





#  LAUNCH SCRIPT
if __name__ == "__main__":
    while True:
        query = listen().lower()
        print(query)

        #  RESPONSE
        response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=query,
                    temperature=0.5,
                    max_tokens=1000,
                    top_p=1.0,
                    frequency_penalty=0.5,
                    presence_penalty=0.0)
        query = response['choices'][0]['text']

        #  SILERO VOICE
        language = 'ru'
        model_id = 'ru_v3'
        sample_rate = 48000
        speaker = 'baya'
        put_accent = True
        put_yo = True
        device = torch.device('cpu')
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                    model='silero_tts',
                                    language=language,
                                    speaker=model_id)
        model.to(device)
        result = model.apply_tts(text=query,
                                 speaker=speaker,
                                 sample_rate=sample_rate,
                                 put_accent=put_accent,
                                 put_yo=put_yo)
        
        #  PLAY AUDIO ANSWER
        print(query)
        sd.play(result, sample_rate)