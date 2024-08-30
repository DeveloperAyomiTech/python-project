from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.authenticators import IAMAuthenticator
from ibm_cloud_sdk_core import ApiException
import json

# Replace with your IBM Watson credentials
api_key = 'your_api_key'
url = 'your_service_url'

# Set up the NLU client
authenticator = IAMAuthenticator(api_key)
nlu = NaturalLanguageUnderstandingV1(
    version='2022-08-01',
    authenticator=authenticator
)
nlu.set_service_url(url)

def detect_emotions(text):
    try:
        # Analyze the text for emotion
        response = nlu.analyze(
            text=text,
            features={'emotion': {}}  # Request emotion analysis
        ).get_result()
        
        # Extract emotions from the response
        emotions = response.get('emotion', {}).get('document', {}).get('emotion', {})
        return emotions

    except ApiException as e:
        print(f"IBM Watson API Error: {e.message}")
        return None

if __name__ == '__main__':
    sample_text = "I am so happy and excited about the new project!"

    emotions = detect_emotions(sample_text)
    if emotions:
        print("Detected Emotions:")
        for emotion, score in emotions.items():
            print(f"{emotion.capitalize()}: {score}")
    else:
        print("Could not analyze emotions.")
