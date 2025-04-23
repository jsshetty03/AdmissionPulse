import streamlit as st
import sqlite3
import hashlib
from improved_sop_rater import ImprovedSOPRater
from sop_generator import generate_sop
from lor_generator import generate_lor
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
import time

# Import suggest_improvements function if possible, otherwise define it here
try:
    from advancedmodel2 import suggest_improvements
except ImportError:
    # Define the function here as a fallback
    # Function to suggest improvements for better admission chances
    def suggest_improvements(input_data, feature_importances, target_rank):
        """
        Suggest improvements based on feature importance and user's current profile
        to increase chances of admission to universities with the target rank.
        
        Args:
            input_data: Dictionary containing the user's current profile
            feature_importances: Dictionary of feature names and their importance values
            target_rank: Desired university rank (1-5, where 5 is highest)
        
        Returns:
            List of personalized suggestions
        """
        suggestions = []
        current_rank = input_data.get('University Rating', 0)
        
        # Sort features by importance
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        
        # If target rank is higher (higher number) than current profile suggests
        if target_rank > current_rank:  # FIXED: Changed to use > for higher-ranked universities
            suggestions.append(f"\n Suggestions to Improve Admission Chances for Rank {target_rank} Universities ")
            
            # Check GRE score
            if 'GRE Score' in input_data:
                gre_score = input_data['GRE Score']
                if gre_score < 320 and target_rank >= 4:
                    suggestions.append(f"• Improve your GRE score (currently {gre_score}). Aim for at least 320+ for rank {target_rank} universities.")
                elif gre_score < 310 and target_rank >= 3:
                    suggestions.append(f"• Improve your GRE score (currently {gre_score}). Aim for at least 310+ for rank {target_rank} universities.")
                elif gre_score < 300 and target_rank >= 2:
                    suggestions.append(f"• Improve your GRE score (currently {gre_score}). Aim for at least 300+ for rank {target_rank} universities.")
            
            # Check TOEFL score
            if 'TOEFL Score' in input_data:
                toefl_score = input_data['TOEFL Score']
                if toefl_score < 105 and target_rank >= 4:
                    suggestions.append(f"• Improve your TOEFL score (currently {toefl_score}). Aim for at least 105+ for rank {target_rank} universities.")
                elif toefl_score < 100 and target_rank >= 3:
                    suggestions.append(f"• Improve your TOEFL score (currently {toefl_score}). Aim for at least 100+ for rank {target_rank} universities.")
                elif toefl_score < 90 and target_rank >= 2:
                    suggestions.append(f"• Improve your TOEFL score (currently {toefl_score}). Aim for at least 90+ for rank {target_rank} universities.")
            
            # Check CGPA
            if 'CGPA' in input_data:
                cgpa = input_data['CGPA']
                if cgpa < 9.0 and target_rank >= 4:
                    suggestions.append(f"• Your CGPA (currently {cgpa}) is below the typical threshold for rank {target_rank} universities. Focus on improving grades in remaining courses.")
                elif cgpa < 8.5 and target_rank >= 3:
                    suggestions.append(f"• Your CGPA (currently {cgpa}) is below the typical threshold for rank {target_rank} universities. Focus on improving grades in remaining courses.")
                elif cgpa < 8.0 and target_rank >= 2:
                    suggestions.append(f"• Your CGPA (currently {cgpa}) is below the typical threshold for rank {target_rank} universities. Focus on improving grades in remaining courses.")
            
            # Check Research
            if 'Research' in input_data:
                research = input_data['Research']
                if research == 0 and target_rank >= 3:
                    suggestions.append("• Consider gaining research experience. Research experience is highly valued by higher-ranked universities.")
                    suggestions.append("  - Try to publish in recognized journals or conferences")
                    suggestions.append("  - Participate in research projects with professors")
                    suggestions.append("  - Complete a research-focused capstone or thesis project")
            
            # Check SOP strength
            if 'SOP' in input_data:
                sop = input_data['SOP']
                if sop < 4.0 and target_rank >= 3:
                    suggestions.append(f"• Strengthen your Statement of Purpose (currently rated {sop}/5.0):")
                    suggestions.append("  - Clearly articulate your research interests and career goals")
                    suggestions.append("  - Highlight specific professors or research groups you want to work with")
                    suggestions.append("  - Explain why this specific university is the right fit for you")
                    suggestions.append("  - Demonstrate how your background prepares you for success in their program")
            
            # Check LOR strength
            if 'LOR' in input_data:
                lor = input_data['LOR']
                if lor < 4.0 and target_rank >= 3:
                    suggestions.append(f"• Obtain stronger Letters of Recommendation (currently rated {lor}/5.0):")
                    suggestions.append("  - Request letters from professors who know you well academically")
                    suggestions.append("  - Consider letters from research supervisors or internship mentors")
                    suggestions.append("  - Provide recommenders with your CV and statement of purpose")
                    suggestions.append("  - Remind them to highlight specific achievements and potential")
            
            # Additional general suggestions based on university ranking
            if target_rank >= 4:
                suggestions.append("\n• Additional ways to strengthen your application for top-ranked universities:")
                suggestions.append("  - Participate in relevant internships at research institutions or industry")
                suggestions.append("  - Win competitive scholarships or academic awards")
                suggestions.append("  - Develop specialized technical skills relevant to your field")
                suggestions.append("  - Network with alumni or faculty from target universities")
                suggestions.append("  - Consider applying for pre-master's research positions at target universities")
            
            # If no specific improvements needed
            if len(suggestions) <= 1:
                suggestions.append("Your profile already meets most requirements for your target university rank!")
                suggestions.append("Focus on crafting an exceptional application that highlights your unique strengths.")
            
        else:
            suggestions.append(f"\nYour current profile is already well-aligned with rank {target_rank} universities.")
            suggestions.append("Focus on a well-crafted application that highlights your specific strengths and fit with each program.")
        
        return suggestions

# Set page configuration at the very beginning
st.set_page_config(
    page_title="AdmissionPulse",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
def add_custom_css():
    st.markdown("""
    <style>
        /* Main app styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1E3A8A;
            font-family: 'Helvetica Neue', sans-serif;
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        h2 {
            font-size: 1.8rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        h3 {
            font-size: 1.5rem;
            font-weight: 500;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
        }
        
        /* Form styling */
        .stForm {
            background-color: #F8FAFC;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #2563EB;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
            border: none;
        }
        
        .stButton > button:hover {
            background-color: #1E40AF;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #E0E7FF;
            border-radius: 5px 5px 0 0;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #3B82F6;
            color: white;
        }
        
        /* Text area styling */
        .stTextArea textarea {
            border: 1px solid #CBD5E1;
            border-radius: 5px;
            padding: 0.5rem;
        }
        
        /* Metrics styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #2563EB;
        }
        
        /* Progress bar styling */
        .stProgress > div > div {
            background-color: #3B82F6;
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 5px;
            overflow: hidden;
            border: 1px solid #E2E8F0;
        }
        
        .dataframe th {
            background-color: #DBEAFE;
            color: #1E40AF;
            font-weight: 600;
            padding: 0.75rem 1rem;
            text-align: left;
        }
        
        .dataframe td {
            padding: 0.75rem 1rem;
            border-top: 1px solid #E2E8F0;
        }
        
        /* Success/info/warning/error message styling */
        .stSuccess, .stInfo, .stWarning, .stError {
            border-radius: 5px;
            padding: 0.75rem 1rem;
            margin: 1rem 0;
        }
        
        /* Sidebar styling */
        .css-6qob1r.e1fqkh3o3 {
            background-color: #F1F5F9;
            padding: 2rem 1rem;
            border-right: 1px solid #E2E8F0;
        }
        
        /* Input fields */
        .stTextInput > div > div > input, .stNumberInput > div > div > input {
            border-radius: 5px;
            border: 1px solid #CBD5E1;
            padding: 0.5rem;
        }
        
        /* Logo animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .logo-animation {
            animation: pulse 2s infinite;
            display: inline-block;
        }
        
        /* Card styling */
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            border: 1px solid #E2E8F0;
        }
        
        /* Divider styling */
        hr {
            margin: 2rem 0;
            border: none;
            height: 1px;
            background-color: #E2E8F0;
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Removed the DROP TABLE statement to maintain persistent user data
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 username TEXT PRIMARY KEY,
                 password TEXT,
                 email TEXT UNIQUE,
                 full_name TEXT)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, email, full_name):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password, email, full_name) VALUES (?,?,?,?)',
                  (username, hashed_password, email, full_name))
        conn.commit()
        print(f"User '{username}' added successfully with hashed password '{hashed_password}'")
        return True
    except sqlite3.IntegrityError as e:
        print(f"Integrity Error: {e}")
        return False
    except sqlite3.Error as e:
        print(f"Database Error: {e}")
        return False
    finally:
        conn.close()

def fetch_user(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=?', (username,))
    user = c.fetchone()
    conn.close()
    print(f"Fetched user data for '{username}': {user}")
    return user

def login_user(username, password):
    user = fetch_user(username)
    if user and user[1] == hash_password(password):  # Ensure passwords match
        return True
    print(f"User '{username}' login failed.")
    return False

# Initialize the database
initialize_db()

# Add the custom CSS
add_custom_css()

def show_animation():
    """Show a simple loading animation"""
    progress_text = "Operation in progress. Please wait..."
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)  # Adjust the speed of the animation
        progress_bar.progress(percent_complete + 1)
    progress_bar.empty()

def main_app():
    # App header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <span class='logo-animation' style='font-size: 3rem;'>🎓</span>
            <h1>AdmissionPulse</h1>
            <p style='font-size: 1.2rem; color: #475569; margin-bottom: 2rem; text-align: center;'>
                Your all-in-one tool for graduate application documents and admission predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create modern-looking tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 SOP Generator", 
        "⭐ Rate Prewritten SOP", 
        "📋 LOR Generator", 
        "🔮 Admission Prediction"
    ])

    def tab_sop():
        st.markdown("""
        <div class='card'>
            <h2 style='text-align: center; margin-bottom: 1.5rem;'>Generate Statement of Purpose</h2>
            <p style='text-align: center; margin-bottom: 1.5rem; color: #475569;'>
                Fill in the form below to generate a customized Statement of Purpose that stands out.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("sop_form"):
            st.markdown("<h3 style='margin-top: 0;'>Personal Information</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                program = st.text_input("Program", placeholder="e.g., Master's in Computer Science")
                university = st.text_input("University", placeholder="e.g., Stanford University")
                field_interest = st.text_input("Field of Interest", placeholder="e.g., Artificial Intelligence")
            with col2:
                career_goal = st.text_input("Career Goal", placeholder="e.g., AI Researcher")
                subjects_studied = st.text_input("Subjects Studied", placeholder="e.g., Machine Learning, Data Science")
            
            st.markdown("<h3>Projects & Experience</h3>", unsafe_allow_html=True)
            projects_internships = st.text_area("Projects/Internships", placeholder="e.g., NLP Chatbot, AI Recommendation System", height=100)
            
            st.markdown("<h3>Personal Growth & University Fit</h3>", unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                lacking_skills = st.text_input("Skills You Want to Develop", placeholder="e.g., Deep Learning Optimization")
                program_benefits = st.text_input("Program Benefits", placeholder="e.g., Advanced AI Research Labs")
            with col4:
                contribution = st.text_input("How You Plan to Contribute", placeholder="e.g., AI Ethics Research")
            
            submit_button_col1, submit_button_col2, submit_button_col3 = st.columns([1, 1, 1])
            with submit_button_col2:
                submit_sop = st.form_submit_button("✨ Generate SOP")

        if submit_sop:
            with st.spinner("Crafting your personalized Statement of Purpose..."):
                show_animation()  # Show custom animation
                responses = {
                    "program": program,
                    "university": university,
                    "field_interest": field_interest,
                    "career_goal": career_goal,
                    "subjects_studied": subjects_studied,
                    "projects_internships": projects_internships,
                    "lacking_skills": lacking_skills,
                    "program_benefits": program_benefits,
                    "contribution": contribution
                }
                try:
                    # Generate the SOP
                    sop = generate_sop(responses)
                    
                    # Validate the SOP
                    if sop is None or not isinstance(sop, str):
                        st.error("Failed to generate SOP. Please check your inputs and try again.")
                        return

                    # Rate the SOP
                    result = ImprovedSOPRater().rate_sop(sop)
                    st.success("SOP successfully generated!")
                    
                    # Display SOP and rating in a nice card
                    st.markdown("""
                    <div class='card'>
                        <h3 style='text-align: center; margin-bottom: 1rem;'>Your Generated Statement of Purpose</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # SOP content in a styled text area
                    st.text_area("SOP Content", sop, height=400)
                    
                    # Rating display
                    col_rating1, col_rating2, col_rating3 = st.columns([1, 1, 1])
                    with col_rating2:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1rem; background-color: #DBEAFE; border-radius: 10px;'>
                            <h3 style='margin-top: 0; margin-bottom: 0.5rem;'>SOP Quality Rating</h3>
                            <div style='font-size: 2.5rem; font-weight: 700; color: #2563EB;'>
                                {result['rating']}/5
                            </div>
                            <div style='margin-top: 0.5rem; color: #475569;'>
                                Confidence: {result['confidence']:.2f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.progress(result['confidence'])
                    
                    # Download button with improved styling
                    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
                    with col_btn2:
                        st.download_button(
                            label="📥 Download SOP as Text",
                            data=sop,
                            file_name=f"generated_sop_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"An error occurred while generating the SOP: {str(e)}")

    def tab_rate_sop():
        st.markdown("""
        <div class='card'>
            <h2 style='text-align: center; margin-bottom: 1.5rem;'>Rate Your Prewritten SOP</h2>
            <p style='text-align: center; margin-bottom: 1.5rem; color: #475569;'>
                Our AI will analyze and rate your existing Statement of Purpose to help you improve it.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Card for SOP input
        st.markdown("""
        <div class='card'>
            <h3 style='margin-top: 0;'>Paste your Statement of Purpose</h3>
        </div>
        """, unsafe_allow_html=True)
        
        prewritten_sop = st.text_area("", placeholder="Paste your SOP here...", height=300)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            rate_button = st.button("📊 Analyze SOP Quality")

        if rate_button:
            if prewritten_sop:
                with st.spinner("Analyzing your Statement of Purpose..."):
                    show_animation()  # Show custom animation
                    result = ImprovedSOPRater().rate_sop(prewritten_sop)
                    
                    # Create card for results
                    st.markdown("""
                    <div class='card'>
                        <h3 style='text-align: center; margin-bottom: 1rem;'>Analysis Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display rating in a visually appealing way
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1.5rem; background-color: #DBEAFE; border-radius: 10px;'>
                            <h3 style='margin-top: 0; margin-bottom: 0.5rem;'>SOP Quality Rating</h3>
                            <div style='font-size: 3rem; font-weight: 700; color: #2563EB;'>
                                {result['rating']}/5
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<h3 style='margin-top: 0;'>Confidence Score</h3>", unsafe_allow_html=True)
                        st.progress(result['confidence'])
                        st.caption(f"Confidence: {result['confidence']:.2f}")
                        
                        # Add interpretation based on the rating
                        if result['rating'] >= 4:
                            st.success("Excellent SOP! Your statement is well-crafted and compelling.")
                        elif result['rating'] >= 3:
                            st.info("Good SOP. Some improvements could make it even stronger.")
                        else:
                            st.warning("Your SOP needs significant improvements to stand out.")
            else:
                st.error("Please paste your SOP first")

    def tab_lor():
        st.markdown("""
        <div class='card'>
            <h2 style='text-align: center; margin-bottom: 1.5rem;'>Generate Letter of Recommendation</h2>
            <p style='text-align: center; margin-bottom: 1.5rem; color: #475569;'>
                Create a professional recommendation letter by filling in the details below.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("lor_form"):
            st.markdown("<h3 style='margin-top: 0;'>Student Information</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                for_whom = st.text_input("Student's Full Name", placeholder="e.g., John Smith")
                how_you_know = st.text_input("Relationship", placeholder="e.g., Professor for 3 years")
            with col2:
                subjects_taught = st.text_input("Subjects Taught", placeholder="e.g., Advanced Machine Learning")
                marks = st.text_input("Academic Performance", placeholder="e.g., Top 5% of the class with an A grade")
            
            st.markdown("<h3>Projects & Achievements</h3>", unsafe_allow_html=True)
            projects = st.text_area("Projects Completed Under Your Supervision", placeholder="e.g., Research on NLP applications, Final year project on data visualization", height=150)
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn2:
                submit_lor = st.form_submit_button("📝 Generate LOR")

            if submit_lor:
                with st.spinner("Crafting the perfect letter of recommendation..."):
                    show_animation()  # Show custom animation
                    responses = {
                        "for_whom": for_whom,
                        "how_you_know": how_you_know,
                        "subjects_taught": subjects_taught,
                        "marks": marks,
                        "projects": projects
                    }
                    try:
                        # Generate the LOR
                        lor = generate_lor(responses)
                    
                        # Validate the LOR
                        if lor is None or not isinstance(lor, str):
                            st.error("Failed to generate LOR. Please check your inputs and try again.")
                            return

                        # Store LOR in session state
                        st.session_state['generated_lor'] = lor
                        st.success("LOR successfully generated!")
                    except Exception as e:
                        st.error(f"An error occurred while generating the LOR: {str(e)}")

        # Display the generated LOR and download button outside the form
        if 'generated_lor' in st.session_state:
            # Create a nice card for the LOR
            st.markdown("""
            <div class='card'>
                <h3 style='text-align: center; margin-bottom: 1rem;'>Your Generated Letter of Recommendation</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the LOR in a styled container
            st.markdown("""
            <div style='background-color: #F8FAFC; padding: 1.5rem; border-radius: 10px; border: 1px solid #E2E8F0;'>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state['generated_lor'])
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add download button with better styling
            col_dwnld1, col_dwnld2, col_dwnld3 = st.columns([1, 1, 1])
            with col_dwnld2:
                st.download_button(
                    label="📥 Download LOR as Text",
                    data=st.session_state['generated_lor'],
                    file_name=f"recommendation_letter_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

    def tab_admission_prediction():
        st.markdown("""
        <div class='card'>
            <h2 style='text-align: center; margin-bottom: 1.5rem;'>Admission Prediction System</h2>
            <p style='text-align: center; margin-bottom: 1.5rem; color: #475569;'>
                Enter your scores and details to predict your chances of admission and get personalized recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        @st.cache_data
        def load_university_data():
            # First try to load from uni_name.csv (the full dataset)
            uni_file_path = 'uni_name.csv'
            if os.path.exists(uni_file_path):
                try:
                    return pd.read_csv(uni_file_path)
                except Exception as e:
                    st.warning(f"Error loading university data from {uni_file_path}: {e}")
            
            # If that fails, try loading from university_data.csv
            if os.path.exists('university_data.csv'):
                try:
                    return pd.read_csv('university_data.csv')
                except Exception as e:
                    st.warning(f"Error loading university data from university_data.csv: {e}")
            
            # If both fail, use a small default dataset
            st.warning("Using default university data. For more options, please ensure uni_name.csv is available.")
            data = {
                'name': ['Stanford University', 'Massachusetts Institute of Technology (MIT)', 
                         'Harvard University', 'University of California--Berkeley', 
                         'Carnegie Mellon University', 'California Institute of Technology (Caltech)',
                         'University of Chicago', 'Princeton University', 'Yale University',
                         'Columbia University'],
                'rating': [5.0, 5.0, 4.8, 4.9, 4.9, 4.8, 4.7, 4.5, 4.8, 4.3],
                'type': ['Private', 'Private', 'Private', 'Public', 'Private', 'Private',
                         'Private', 'Private', 'Private', 'Private'],
                'state': ['CA', 'MA', 'MA', 'CA', 'PA', 'CA', 'IL', 'NJ', 'CT', 'NY']
            }
            df = pd.DataFrame(data)
            # Save this default data to university_data.csv for future use
            try:
                df.to_csv('university_data.csv', index=False)
            except Exception as e:
                st.warning(f"Could not save default university data: {e}")
            return df

        @st.cache_resource
        def load_model():
            try:
                return joblib.load('admission_prediction_model.pkl')
            except Exception as e:
                st.warning(f"Model not found: {e}. Using a placeholder model.")
                # In a real implementation, you might want to implement a fallback model here
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    # This is a placeholder - in real use, you would train this model on your data
                    return model
                except Exception as ex:
                    st.error(f"Failed to create placeholder model: {ex}")
                    return None

        # Use a card layout for the admission prediction form
        st.markdown("""
        <div class='card'>
            <h3 style='margin-top: 0; margin-bottom: 1rem;'>Your Profile Details</h3>
        </div>
        """, unsafe_allow_html=True)

        with st.form("admission_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h4 style='color: #3B82F6; margin-top: 0;'>Test Scores</h4>", unsafe_allow_html=True)
                gre = st.slider("GRE Score", 260, 340, 310, 
                                help="Graduate Record Examination score (260-340)")
                toefl = st.slider("TOEFL Score", 0, 120, 100, 
                                 help="Test of English as a Foreign Language score (0-120)")
                
                st.markdown("<h4 style='color: #3B82F6;'>Academic Profile</h4>", unsafe_allow_html=True)
                cgpa = st.slider("CGPA", 0.0, 10.0, 8.0, 0.1, 
                                help="Cumulative Grade Point Average on a scale of 10")
                research = st.radio("Research Experience", ["No", "Yes"],
                                   help="Whether you have published research papers")
                research_value = 1 if research == "Yes" else 0
                
            with col2:
                st.markdown("<h4 style='color: #3B82F6; margin-top: 0;'>University & Application</h4>", unsafe_allow_html=True)
                uni_rating = st.slider("Current University Rating", 1, 5, 3, 
                                      help="Rating of your current university (1-5, where 5 is highest)")
                sop_score = st.slider("SOP Strength", 1.0, 5.0, 3.5, 0.1,
                                     help="Estimated strength of your Statement of Purpose (1-5)")
                lor_score = st.slider("LOR Strength", 1.0, 5.0, 3.5, 0.1,
                                     help="Estimated strength of your Letters of Recommendation (1-5)")
                
                # Add target university rank field with better styling
                st.markdown("<h4 style='color: #3B82F6;'>Get Personalized Suggestions</h4>", unsafe_allow_html=True)
                target_rank = st.slider("Target University Rank", 1, 5, 3, 
                                       help="Desired university ranking (1-5, where 5 is highest)")
                
            # Center the submit button
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn2:
                submit_prediction = st.form_submit_button("🔮 Predict Admission Chances")

            if submit_prediction:
                # Show a loading animation
                with st.spinner("Analyzing your profile and calculating admission probabilities..."):
                    show_animation()
                    
                    # Load the university data
                    uni_data = load_university_data()
                    
                    # Check if university data was loaded successfully
                    if uni_data is None or len(uni_data) == 0:
                        st.error("Failed to load university data. Please check your data files.")
                        return
                    
                    # Load the model
                    model = load_model()
                    if model is None:
                        st.error("Failed to load prediction model. Please check your model file.")
                        return
                    
                    # Prepare the input data
                    input_data = {
                        'GRE Score': gre,
                        'TOEFL Score': toefl,
                        'University Rating': uni_rating,
                        'SOP': sop_score,
                        'LOR': lor_score,
                        'CGPA': cgpa,
                        'Research': research_value
                    }
                    
                    # Make the prediction
                    try:
                        prediction = model.predict(pd.DataFrame([input_data]))[0]
                        prediction = max(0, min(1, prediction))  # Ensure the prediction is between 0 and 1
                        
                        # Create a card for prediction results
                        st.markdown("""
                        <div class='card'>
                            <h3 style='text-align: center; margin-top: 0; margin-bottom: 1rem;'>Your Admission Prediction Results</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create a visually appealing prediction display
                        col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
                        with col_pred2:
                            # Determine color based on prediction value
                            color = "#059669" if prediction >= 0.7 else "#D97706" if prediction >= 0.4 else "#DC2626"
                            
                            st.markdown(f"""
                            <div style='text-align: center; padding: 2rem; background-color: #F8FAFC; border-radius: 10px; border: 1px solid #E2E8F0;'>
                                <h3 style='margin-top: 0; margin-bottom: 0.5rem;'>Predicted Chance of Admission</h3>
                                <div style='font-size: 3.5rem; font-weight: 700; color: {color};'>
                                    {prediction * 100:.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show progress bar
                        st.progress(prediction)
                        
                        # Provide interpretation with icons
                        if prediction >= 0.8:
                            st.success("🌟 Very high chance of admission! Your profile is excellent.")
                        elif prediction >= 0.6:
                            st.info("👍 Good chance of admission. You have a competitive profile.")
                        elif prediction >= 0.4:
                            st.warning("⚠️ Moderate chance. Consider strengthening your application.")
                        else:
                            st.error("⚠️ Lower chance of admission. Consider additional preparation or different programs.")
                        
                        # Create a card for university recommendations
                        st.markdown("""
                        <div class='card'>
                            <h3 style='text-align: center; margin-top: 0; margin-bottom: 1rem;'>Recommended Universities</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Find recommended universities based on the selected rating
                        try:
                            # Calculate the difference between the selected rating and each university's rating
                            uni_data['rating_diff'] = abs(uni_data['rating'] - uni_rating)
                            
                            # Sort by the difference and get the top 5
                            recommended_unis = uni_data.sort_values(by='rating_diff').head(5)
                            
                            # Display the recommended universities
                            st.dataframe(recommended_unis[['name', 'rating', 'type', 'state']], hide_index=True, column_config={
                                "name": st.column_config.TextColumn("University Name", width="large"),
                                "rating": st.column_config.NumberColumn("Rating", format="%.1f"),
                                "type": st.column_config.TextColumn("Type", width="small"),
                                "state": st.column_config.TextColumn("State", width="small")
                            })
                        except Exception as e:
                            st.error(f"Error recommending universities: {e}")
                        
                        # Generate improvement suggestions based on target university rank
                        if target_rank != uni_rating:
                            try:
                                # Create a card for improvement suggestions
                                st.markdown("""
                                <div class='card'>
                                    <h3 style='text-align: center; margin-top: 0; margin-bottom: 1rem;'>Personalized Improvement Plan</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Get feature importances from the model if available
                                feature_importances = {}
                                if hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'feature_importances_'):
                                    importances = model.named_steps['model'].feature_importances_
                                    for i, feature in enumerate(model.feature_names_in_):
                                        feature_importances[feature] = importances[i]
                                elif hasattr(model, 'feature_importances_'):
                                    # For standalone models like RandomForest
                                    importances = model.feature_importances_
                                    features = list(input_data.keys())
                                    for i, feature in enumerate(features):
                                        feature_importances[feature] = importances[i]
                                else:
                                    # Use default importances
                                    feature_importances = {
                                        'GRE Score': 0.15,
                                        'TOEFL Score': 0.12,
                                        'University Rating': 0.10,
                                        'SOP': 0.12,
                                        'LOR': 0.12,
                                        'CGPA': 0.25,
                                        'Research': 0.14
                                    }
                                
                                # Generate and display suggestions
                                suggestions = suggest_improvements(input_data, feature_importances, target_rank)
                                
                                # Display suggestions in a styled container
                                st.markdown("""
                                <div style='background-color: #F8FAFC; padding: 1.5rem; border-radius: 10px; border: 1px solid #E2E8F0;'>
                                """, unsafe_allow_html=True)
                                
                                for suggestion in suggestions:
                                    if suggestion.startswith('==='):
                                        st.markdown(f"<h4 style='color: #3B82F6;'>{suggestion.strip('=').strip()}</h4>", unsafe_allow_html=True)
                                    elif suggestion.startswith('•'):
                                        st.markdown(f"<p style='margin-bottom: 0.5rem;'>{suggestion}</p>", unsafe_allow_html=True)
                                    elif suggestion.startswith('  -'):
                                        st.markdown(f"<p style='margin-left: 2rem; margin-bottom: 0.25rem; color: #475569;'>{suggestion}</p>", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"<p>{suggestion}</p>", unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Calculate improved prediction if pursuing higher-ranked university
                                if target_rank < uni_rating:
                                    improved_data = input_data.copy()
                                    improved_data['University Rating'] = target_rank
                                    improved_df = pd.DataFrame([improved_data])
                                    improved_prediction = model.predict(improved_df)[0]
                                    improved_prediction = max(0, min(1, improved_prediction))
                                    
                                    # Display prediction with improvements
                                    col_imp1, col_imp2, col_imp3 = st.columns([1, 1, 1])
                                    with col_imp2:
                                        st.markdown(f"""
                                        <div style='text-align: center; padding: 1.5rem; background-color: #DBEAFE; border-radius: 10px; border: 1px solid #93C5FD;'>
                                            <h4 style='margin-top: 0; margin-bottom: 0.5rem;'>Probability with Improvements</h4>
                                            <div style='font-size: 2.5rem; font-weight: 700; color: #2563EB;'>
                                                {improved_prediction * 100:.1f}%
                                            </div>
                                            <div style='color: {("#059669" if improved_prediction > prediction else "#DC2626")}; font-weight: 500;'>
                                                {(improved_prediction - prediction) * 100:+.1f}%
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Error generating improvement suggestions: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")

    with tab1:
        tab_sop()
    with tab2:
        tab_rate_sop()
    with tab3:
        tab_lor()
    with tab4:
        tab_admission_prediction()

    # Create modern sidebar with better formatting
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <span style='font-size: 2.5rem;'>🎓</span>
            <h2 style='margin-top: 0.5rem;'>AdmissionPulse</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Current date display
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 1.5rem;'>
            <p style='color: #64748B; font-size: 0.9rem;'>
                {datetime.now().strftime('%A, %B %d, %Y')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # About section with better formatting
        st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
            <h3 style='margin-top: 0; color: #1E3A8A;'>About</h3>
            <p style='margin-bottom: 0.75rem;'>
                AdmissionPulse helps students navigate the graduate application process with:
            </p>
            <ul style='margin-bottom: 0; padding-left: 1.5rem;'>
                <li>Professional Statements of Purpose</li>
                <li>Expert SOP quality assessment</li>
                <li>Custom Letters of Recommendation</li>
                <li>AI-powered admission predictions</li>
                <li>Personalized improvement suggestions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # How to use section with icons
        st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
            <h3 style='margin-top: 0; color: #1E3A8A;'>How to use</h3>
            <ol style='padding-left: 1.5rem; margin-bottom: 0;'>
                <li><strong>Select a tab</strong> for your desired function</li>
                <li><strong>Fill in</strong> the required information</li>
                <li><strong>Submit</strong> the form to get results</li>
                <li><strong>Download</strong> generated documents</li>
                <li><strong>Get improvement tips</strong> for better admission chances</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Add supportive statistics
        st.markdown("""
        <div style='margin-top: 2rem; text-align: center;'>
            <h4 style='color: #1E3A8A; margin-bottom: 1rem;'>Why Students Trust Us</h4>
            <div style='display: flex; justify-content: space-between;'>
                <div style='text-align: center;'>
                    <div style='font-size: 1.5rem; font-weight: 700; color: #2563EB;'>95%</div>
                    <div style='font-size: 0.8rem; color: #64748B;'>Satisfaction</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 1.5rem; font-weight: 700; color: #2563EB;'>10k+</div>
                    <div style='font-size: 0.8rem; color: #64748B;'>Documents</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 1.5rem; font-weight: 700; color: #2563EB;'>87%</div>
                    <div style='font-size: 0.8rem; color: #64748B;'>Admission Rate</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def login_page():
    # Apply a clean, modern login interface
    st.markdown("""
    <style>
        div.stButton > button {
            width: 100%;
            background-color: #2563EB;
            color: white;
            border: none;
            padding: 0.75rem 0;
            font-weight: 500;
            border-radius: 5px;
            margin-top: 1rem;
        }
        div.stButton > button:hover {
            background-color: #1E40AF;
        }
    </style>
    """, unsafe_allow_html=True)

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Logo and title
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <span style='font-size: 4rem;' class='logo-animation'>🎓</span>
            <h1 style='margin-top: 0.5rem;'>AdmissionPulse</h1>
            <p style='color: #475569; font-size: 1.2rem;'>Your graduate admission companion</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Login card
        st.markdown("""
        <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;'>
            <h2 style='margin-top: 0; margin-bottom: 1.5rem; text-align: center; color: #1E3A8A;'>Login to Your Account</h2>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        login_button = st.button("Login")
        
        st.markdown("<p style='text-align: center; margin-top: 1rem;'>Don't have an account?</p>", unsafe_allow_html=True)
        signup_button = st.button("Create New Account")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add testimonials
        st.markdown("""
        <div style='margin-top: 2rem;'>
            <h3 style='text-align: center; color: #1E3A8A; margin-bottom: 1rem;'>What Students Say</h3>
            <div style='background-color: #F8FAFC; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #3B82F6;'>
                <p style='font-style: italic; margin-bottom: 0.5rem;'>"AdmissionPulse helped me craft the perfect SOP that got me into IIT CHICAGO CS program!"</p>
                <p style='text-align: right; margin-bottom: 0; color: #64748B;'>— AMMAR D., IIT CHICAGO University</p>
            </div>
            <div style='background-color: #F8FAFC; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3B82F6;'>
                <p style='font-style: italic; margin-bottom: 0.5rem;'>"The admission prediction was spot on. I followed the suggestions and got into my dream university."</p>
                <p style='text-align: right; margin-bottom: 0; color: #64748B;'>— VARANASYA K., George Mason</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if login_button:
        if login_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.rerun()
        else:
            st.error("Invalid username or password")

    if signup_button:
        st.session_state['show_signup'] = True
        st.rerun()

def signup_page():
    # Apply a clean, modern signup interface
    st.markdown("""
    <style>
        div.stButton > button {
            width: 100%;
            background-color: #2563EB;
            color: white;
            border: none;
            padding: 0.75rem 0;
            font-weight: 500;
            border-radius: 5px;
            margin-top: 1rem;
        }
        div.stButton > button:hover {
            background-color: #1E40AF;
        }
        .secondary-button > button {
            background-color: #E2E8F0;
            color: #1E293B;
        }
        .secondary-button > button:hover {
            background-color: #CBD5E1;
        }
    </style>
    """, unsafe_allow_html=True)

    # Center the signup form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Logo and title
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <span style='font-size: 4rem;' class='logo-animation'>🎓</span>
            <h1 style='margin-top: 0.5rem;'>AdmissionPulse</h1>
            <p style='color: #475569; font-size: 1.2rem;'>Your graduate admission companion</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Signup card
        st.markdown("""
        <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <h2 style='margin-top: 0; margin-bottom: 1.5rem; text-align: center; color: #1E3A8A;'>Create Your Account</h2>
        """, unsafe_allow_html=True)
        
        new_username = st.text_input("Username", placeholder="Choose a username")
        new_email = st.text_input("Email Address", placeholder="Enter your email")
        full_name = st.text_input("Full Name", placeholder="Enter your full name")
        
        col1, col2 = st.columns(2)
        with col1:
            new_password = st.text_input("Password", type="password", placeholder="Create password")
        with col2:
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
        
        signup_button = st.button("Create Account")
        
        st.markdown("<div class='secondary-button'>", unsafe_allow_html=True)
        back_button = st.button("Back to Login")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add benefits section
        st.markdown("""
        <div style='margin-top: 2rem;'>
            <h3 style='text-align: center; color: #1E3A8A; margin-bottom: 1rem;'>Benefits of Joining</h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
                <div style='background-color: #F8FAFC; padding: 1rem; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>💾</div>
                    <h4 style='margin-top: 0; margin-bottom: 0.5rem; color: #1E3A8A;'>Save Your Documents</h4>
                    <p style='margin-bottom: 0; font-size: 0.9rem; color: #475569;'>Store all your application materials in one place</p>
                </div>
                <div style='background-color: #F8FAFC; padding: 1rem; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>📊</div>
                    <h4 style='margin-top: 0; margin-bottom: 0.5rem; color: #1E3A8A;'>Track Progress</h4>
                    <p style='margin-bottom: 0; font-size: 0.9rem; color: #475569;'>Monitor your application journey</p>
                </div>
                <div style='background-color: #F8FAFC; padding: 1rem; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>🔔</div>
                    <h4 style='margin-top: 0; margin-bottom: 0.5rem; color: #1E3A8A;'>Get Updates</h4>
                    <p style='margin-bottom: 0; font-size: 0.9rem; color: #475569;'>Receive deadline reminders and tips</p>
                </div>
                <div style='background-color: #F8FAFC; padding: 1rem; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>🔒</div>
                    <h4 style='margin-top: 0; margin-bottom: 0.5rem; color: #1E3A8A;'>Secure Access</h4>
                    <p style='margin-bottom: 0; font-size: 0.9rem; color: #475569;'>Keep your information private and secure</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if signup_button:
        if not new_username or not new_email or not full_name or not new_password:
            st.error("Please fill in all fields")
        elif new_password != confirm_password:
            st.error("Passwords do not match")
        else:
            success = create_user(new_username, new_password, new_email, full_name)
            if success:
                st.success("Account created successfully!")
                st.info("You can now log in to your account.")
                
                # Add a delay to show the success message
                time.sleep(1.5)
                
                st.session_state['show_signup'] = False
                st.rerun()
            else:
                st.error("Username or email already exists")

    if back_button:
        st.session_state['show_signup'] = False
        st.rerun()

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None

def app():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'show_signup' not in st.session_state:
        st.session_state['show_signup'] = False

    if not st.session_state['logged_in']:
        if st.session_state['show_signup']:
            signup_page()
        else:
            login_page()
    else:
        # Welcome user in sidebar
        st.sidebar.markdown(f"""
        <div style='margin-bottom: 1rem; padding: 1rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
            <p style='margin-bottom: 0.5rem; font-size: 1.2rem; font-weight: 600; color: #1E3A8A;'>
                Welcome, {st.session_state.get('username', 'User')}! 👋
            </p>
            <p style='margin-bottom: 0; color: #64748B; font-size: 0.9rem;'>
                Last login: {datetime.now().strftime('%B %d, %Y')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add logout button with better styling
        st.sidebar.markdown("""
        <style>
            .logout-button > button {
                width: 100%;
                background-color: #F1F5F9;
                color: #64748B;
                border: 1px solid #CBD5E1;
                font-weight: 500;
            }
            .logout-button > button:hover {
                background-color: #E2E8F0;
                color: #475569;
            }
        </style>
        <div class='logout-button'>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("Sign Out"):
            logout()
            st.rerun()
            
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        
        # Main application
        main_app()

if __name__ == "__main__":
    app()
