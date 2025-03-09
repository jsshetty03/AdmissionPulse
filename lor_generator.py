import requests
import json
from config import OPENROUTER_API_KEY

def generate_lor(responses):
    prompt = f"""
    Write a formal and professional Letter of Recommendation (LOR) for a student. 
    Ensure the LOR is **at least 500 words** and follows a formal academic tone.

    Use the following details:
    - For Whom the Letter Is: {responses['for_whom']}
    - How You Know the Student: {responses['how_you_know']}
    - Subjects Taught: {responses['subjects_taught']}
    - Marks in Your Subject: {responses['marks']}
    - Projects Completed Under You: {responses['projects']}

    The LOR should have the following structure:
    1. **Introduction**: My first observation of [Student Name] was...
    2. **Body**: Discuss the student's academic performance, projects, and skills.
    3. **Conclusion**: Strongly recommend the student for the program.

    The final LOR should be engaging, persuasive, and free from grammatical errors.
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-3.5-turbo",  # Use GPT-3.5 or GPT-4
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        response_json = response.json()
        lor = response_json["choices"][0]["message"]["content"]
        return lor
    
    except Exception as e:
        print(f"Error generating LOR: {e}")
        return "Failed to generate LOR. Please try again later."