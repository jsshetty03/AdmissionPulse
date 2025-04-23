import requests
import json
from config import OPENROUTER_API_KEY  # Ensure the API key is correct

def generate_sop(responses):
    """
    Generates a well-structured SOP using OpenRouter's Gemini API.
    """
    prompt = f"""
    Write a well-structured, properly formatted Statement of Purpose (SOP) for a university application. 
    Ensure the SOP is **at least 672 words** and follows a **formal academic tone**.

    **Structure:**
    **Introduction**: Discuss motivation for choosing this program.
    **Academic Background**: Mention relevant subjects, coursework, and performance.
    **Projects & Experience**: Highlight internships, projects, and skills developed.
    **Career Goals**: Explain long-term vision and why this program is crucial.
    **Conclusion**: Summarize and reinforce enthusiasm for the program.

    **User Details:**
    - **Program**: {responses['program']}
    - **University**: {responses['university']}
    - **Field of Interest**: {responses['field_interest']}
    - **Career Goal**: {responses['career_goal']}
    - **Subjects Studied**: {responses['subjects_studied']}
    - **Projects/Internships**: {responses['projects_internships']}
    - **Lacking Skills**: {responses['lacking_skills']}
    - **Program Benefits**: {responses['program_benefits']}
    - **Contribution to the Field**: {responses['contribution']}
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "google/gemini-2.0-flash-lite-001",  # ✅ Ensure model name is correct
        "messages": [{"role": "user", "content": prompt}],# OpenRouter uses "messages"
        "temperature": 0.7,  # Add temperature for more creative outputs
        "top_p": 0.9  # Add top_p for better text quality
    }

    try:
        print("🔄 Sending request to OpenRouter's Gemini API...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )

        print("📜 Raw API Response:", response.text)  # Debugging API response

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")

        response_json = response.json()

        # ✅ Extract generated text properly
        if "choices" in response_json and len(response_json["choices"]) > 0:
            sop = response_json["choices"][0]["message"]["content"]
            return sop
        else:
            raise Exception(f"Unexpected API response format: {response_json}")

    except Exception as e:
        print(f"❌ Error generating SOP: {e}")
        return "Failed to generate SOP. Please try again later."


# **✅ Test Case**
if __name__ == "__main__":
    user_data = {
        "program": "Master's in Computer Science",
        "university": "Stanford University",
        "field_interest": "Artificial Intelligence",
        "career_goal": "AI Researcher",
        "subjects_studied": "Machine Learning, Data Science",
        "projects_internships": "NLP-based Chatbot, AI-driven Recommendation System",
        "lacking_skills": "Deep Learning Optimization",
        "program_benefits": "Advanced AI Research Labs, Industry Collaboration",
        "contribution": "AI Ethics Research, Open-Source AI Projects"
    }

    generated_sop = generate_sop(user_data)
    print("\n🔹 **Generated SOP:**\n", generated_sop)
