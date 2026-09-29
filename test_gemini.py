import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=key)
try:
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents='Return a JSON with key "status" and value "ok"',
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type='application/json'
        )
    )
    print('Response:', response.text)
except Exception as e:
    print('Error:', e)
