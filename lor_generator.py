import requests
import json
from config import OPENROUTER_API_KEY

def generate_lor(data):
    """
    Generates a well-structured Letter of Recommendation using OpenRouter's Gemini API.
    """
    prompt = f"""
    Write a formal Letter of Recommendation for a student applying to graduate school.
    
    **Details:**
    - **Student Name**: {data['for_whom']}
    - **Your Relationship**: {data['how_you_know']}
    - **Subjects Taught**: {data['subjects_taught']}
    - **Academic Performance**: {data['marks']}
    - **Projects Completed**: {data['projects']}
    
    The letter should be formal, detailed, and highlight the student's academic achievements, 
    personal qualities, and potential for success in graduate studies.
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "google/gemini-2.0-flash-lite-001", 
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9
    }

    try:
        print("🔄 Sending request to OpenRouter's Gemma 3 API...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")

        response_json = response.json()

        if "choices" in response_json and len(response_json["choices"]) > 0:
            lor = response_json["choices"][0]["message"]["content"]
            return lor
        else:
            raise Exception(f"Unexpected API response format: {response_json}")

    except Exception as e:
        print(f"❌ Error generating LOR: {e}")
        return "Failed to generate Letter of Recommendation. Please try again later."
