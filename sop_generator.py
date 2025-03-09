import requests
import json
from config import OPENROUTER_API_KEY

def generate_sop(responses):
    prompt = f"""
    Write a detailed and well-structured Statement of Purpose (SOP) for a university application. 
    Ensure the SOP is **at least 672 words** and follows a formal academic tone.
    
    Use the following details:
    - Program: {responses['program']}
    - University: {responses['university']}
    - Field Interest: {responses['field_interest']}
    - Career Goal: {responses['career_goal']}
    - Subjects Studied: {responses['subjects_studied']}
    - Projects/Internships: {responses['projects_internships']}
    - Lacking Skills: {responses['lacking_skills']}
    - Program Benefits: {responses['program_benefits']}
    - Contribution: {responses['contribution']}
    
    The SOP should have **clear paragraphs**, a strong introduction, a well-defined body, and a compelling conclusion. 
    Highlight motivation, academic background, professional experience, and how this program aligns with career goals.
    
    The final SOP should be engaging, persuasive, and free from grammatical errors.
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-3.5-turbo",  # Try a different model if gpt-4 fails
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500
    }

    try:
        print("Sending request to OpenRouter API...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        # Print the full response for debugging
        print("API Response Status Code:", response.status_code)
        print("API Response Body:", response.text)
        
        # Check if the request was successful
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        # Parse the response JSON
        response_json = response.json()
        
        # Check if the response contains the expected keys
        if "choices" not in response_json or len(response_json["choices"]) == 0:
            raise Exception("Invalid API response: 'choices' key not found or empty")
        
        # Extract the generated SOP
        sop = response_json["choices"][0]["message"]["content"]
        return sop
    
    except Exception as e:
        # Handle any errors that occur during the API request
        print(f"Error generating SOP: {e}")
        return "Failed to generate SOP. Please try again later."
def postprocess_sop(sop):
    """Improve SOP structure and coherence"""
    # Split into paragraphs and ensure proper formatting
    paragraphs = [p.strip() for p in sop.split('\n') if p.strip()]
    
    # Ensure paragraph structure
    processed = []
    for p in paragraphs:
        if not p.endswith('.'):
            p += '.'
        if len(p.split()) < 15:  # Merge short paragraphs
            if processed:
                processed[-1] += ' ' + p
                continue
        processed.append(p)
    
    # Ensure academic tone
    academic_keywords = ['furthermore', 'moreover', 'consequently', 'notably']
    for i, p in enumerate(processed):
        if i > 0 and not any(kw in p.lower() for kw in academic_keywords):
            processed[i] = 'Furthermore, ' + p[0].lower() + p[1:]
    
    return '\n\n'.join(processed)