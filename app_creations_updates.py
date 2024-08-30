from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.authenticators import IAMAuthenticator
from ibm_cloud_sdk_core import ApiException
import os
from typing import Optional, Dict

# Function to create the Watson NLU client
def create_nlu_client(api_key: str, url: str) -> NaturalLanguageUnderstandingV1:
    authenticator = IAMAuthenticator(api_key)
    nlu = NaturalLanguageUnderstandingV1(
        version='2022-08-01',
        authenticator=authenticator
    )
    nlu.set_service_url(url)
    return nlu

# Function to detect emotions in text
def detect_emotions(api_key: str, url: str, text: str) -> Optional[Dict[str, float]]:
    nlu = create_nlu_client(api_key, url)
    try:
        response = nlu.analyze(
            text=text,
            features={'emotion': {}}
        ).get_result()

        emotions = response.get('emotion', {}).get('document', {}).get('emotion', {})
        return emotions

    except ApiException as e:
        print(f"IBM Watson API Error: {e.message}")
        return None

# Function to format and display the emotion detection results
def format_emotion_output(emotions: Optional[Dict[str, float]]) -> str:
    if emotions is None:
        return "Error in detecting emotions."

    if not emotions:
        return "No emotions detected."

    formatted_output = "Detected Emotions:\n"
    for emotion, score in emotions.items():
        formatted_output += f"{emotion.capitalize()}: {score:.2f}\n"

    return formatted_output

# Main function to execute the application
def main():
    # Replace with your actual API key and URL or set them as environment variables
    api_key = os.getenv('WATSON_API_KEY')
    url = os.getenv('WATSON_URL')

    if not api_key or not url:
        print("Please set the WATSON_API_KEY and WATSON_URL environment variables.")
        return

    sample_text = "I am so happy and excited about the new project!"

    emotions = detect_emotions(api_key, url, sample_text)
    result = format_emotion_output(emotions)
    print(result)

if __name__ == '__main__':
    main()
