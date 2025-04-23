from transformers import pipeline

def rate_sop(sop_text):
    # Load a pre-trained BERT model for text classification
    classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
    
    # Get the rating from the model
    result = classifier(sop_text)
    rating = int(result[0]['label'][0])  # Extract numeric rating
    confidence = result[0]['score']  # Confidence score
    
    return rating, confidence

# Load BERT-based model for SOP rating
rating_pipeline = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

def generate_sop(data):
    sop_template = f"""
    Statement of Purpose

    {data['university']}
    Program: {data['program']}

    From an early age, my fascination with {data['field_interest']} has driven my academic and professional aspirations. My foundational knowledge stems from my studies in {data['subjects_studied']}, which provided me with a solid technical background. Throughout my academic journey, I have undertaken various projects and internships, such as {data['projects_internships']}, allowing me to gain hands-on experience and deepen my understanding.

    Recognizing gaps in my knowledge, particularly in {data['lacking_skills']}, I am motivated to pursue advanced studies in {data['program']} at {data['university']}. This program will equip me with the expertise and research opportunities necessary for my growth and contributions to the field.

    I am particularly drawn to {data['program_benefits']}, as they align with my aspirations of {data['career_goal']}. The opportunity to collaborate with esteemed faculty and peers will be invaluable. Furthermore, I am eager to contribute through {data['contribution']} and make a lasting impact.

    I strongly believe that my background, coupled with my passion for {data['field_interest']}, will enable me to thrive in your program. I look forward to the opportunity to contribute to and benefit from the academic environment at {data['university']}.

    Sincerely,
    [Your Name]
    """

    # Rate the SOP
    sop_rating = rate_sop(sop_template)

    return sop_template, sop_rating

# Example Usage
user_data = {
    "university": "Stanford University",
    "program": "Master’s in Computer Science",
    "field_interest": "Artificial Intelligence",
    "subjects_studied": "Data Structures, Algorithms, and Machine Learning",
    "projects_internships": "AI-powered chatbot, Google AI Internship",
    "lacking_skills": "Deep Learning optimization",
    "program_benefits": "research opportunities and industry connections",
    "career_goal": "becoming an AI Research Scientist",
    "contribution": "innovative AI-based projects and publications"
}

sop_text, sop_rating = generate_sop(user_data)
print(sop_text)
print(f"\nSOP Rating: {sop_rating}/5")
