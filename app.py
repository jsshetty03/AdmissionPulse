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
            suggestions.append(f"\n=== Suggestions to Improve Admission Chances for Rank {target_rank} Universities ===")
            
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
    layout="wide"
)

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

initialize_db()  # Initialize the database without dropping tables

def main_app():
    st.title("AdmissionPulse")
    st.markdown("Your all-in-one tool for graduate application documents and admission predictions")

    tab1, tab2, tab3, tab4 = st.tabs(["SOP Generator", "Rate Prewritten SOP", "LOR Generator", "Admission Prediction"])

    def tab_sop():
        st.header("Generate Statement of Purpose")
        st.markdown("Fill in the form below to generate a customized SOP")

        with st.form("sop_form"):
            col1, col2 = st.columns(2)
            with col1:
                program = st.text_input("Program", "e.g., Master's in Computer Science")
                university = st.text_input("University", "e.g., Stanford University")
                field_interest = st.text_input("Field of Interest", "e.g., Artificial Intelligence")
                career_goal = st.text_input("Career Goal", "e.g., AI Researcher")
                subjects_studied = st.text_input("Subjects Studied", "e.g., Machine Learning, Data Science")
            with col2:
                projects_internships = st.text_area("Projects/Internships", "e.g., NLP Chatbot, AI Recommendation System")
                lacking_skills = st.text_input("Skills You Want to Develop", "e.g., Deep Learning Optimization")
                program_benefits = st.text_input("Program Benefits", "e.g., Advanced AI Research Labs")
                contribution = st.text_input("How You Plan to Contribute", "e.g., AI Ethics Research")
            submit_sop = st.form_submit_button("Generate SOP")

        if submit_sop:
            with st.spinner("Generating your SOP..."):
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
                    st.subheader("Your Generated SOP")
                    st.text_area("SOP Content", sop, height=400)
                    st.metric("SOP Rating", f"{result['rating']}/5")
                    st.progress(result['confidence'])
                    st.caption(f"Confidence: {result['confidence']:.2f}")

                    # Add download button only if SOP is valid
                    st.download_button(
                        label="Download SOP as Text",
                        data=sop,
                        file_name="generated_sop.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"An error occurred while generating the SOP: {str(e)}")

    def tab_rate_sop():
        st.header("Rate Prewritten SOP")
        st.markdown("Paste your prewritten SOP below to get a rating")

        prewritten_sop = st.text_area("Paste your SOP here", height=300)

        if st.button("Rate SOP"):
            if prewritten_sop:
                with st.spinner("Analyzing your SOP..."):
                    result = ImprovedSOPRater().rate_sop(prewritten_sop)
                    st.success("Analysis complete!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("SOP Rating", f"{result['rating']}/5")
                    with col2:
                        st.progress(result['confidence'])
                        st.caption(f"Confidence: {result['confidence']:.2f}")
            else:
                st.error("Please paste your SOP first")

    def tab_lor():
        st.header("Generate Letter of Recommendation")
        st.markdown("Fill in the details below to generate a recommendation letter")

        with st.form("lor_form"):
            for_whom = st.text_input("For Whom the Letter Is", "Student's Full Name")
            how_you_know = st.text_input("How You Know the Student", "e.g., as their professor for 3 years")
            subjects_taught = st.text_input("Subjects Taught", "e.g., Advanced Machine Learning")
            marks = st.text_input("Academic Performance", "e.g., top 5% of the class with an A grade")
            projects = st.text_area("Projects Completed Under Your Supervision", "e.g., Research on NLP applications")
            submit_lor = st.form_submit_button("Generate LOR")

            if submit_lor:
                with st.spinner("Generating your LOR..."):
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
            st.subheader("Your Generated Letter of Recommendation")
            st.markdown(st.session_state['generated_lor'])
        
        # Add download button outside the form
            st.download_button(
                label="Download LOR as Text",
                data=st.session_state['generated_lor'],
                file_name="generated_lor.txt",
                mime="text/plain"
            )

    def tab_admission_prediction():
        st.header("Admission Prediction System")
        st.markdown("Enter your scores and details to predict your chances of admission")

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

        with st.form("admission_form"):
            col1, col2 = st.columns(2)
            with col1:
                gre = st.slider("GRE Score", 260, 340, 310)
                toefl = st.slider("TOEFL Score", 0, 120, 100)
                uni_rating = st.slider("University Rating", 1, 5, 3)
                sop_score = st.slider("SOP Strength", 1.0, 5.0, 3.5, 0.1)
            with col2:
                lor_score = st.slider("LOR Strength", 1.0, 5.0, 3.5, 0.1)
                cgpa = st.slider("CGPA", 0.0, 10.0, 8.0, 0.1)
                research = st.radio("Research Experience", ["No", "Yes"])
                research_value = 1 if research == "Yes" else 0
                
                # Add target university rank field
                st.write("Optional: Get improvement suggestions")
                target_rank = st.slider("Target University Rank (1-5, where 5 is highest)", 1, 5, 2)
                
            submit_prediction = st.form_submit_button("Predict Admission")

            if submit_prediction:
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
                    
                    # Show the prediction results
                    st.success("Prediction complete!")
                    st.subheader("Predicted Chance of Admission")
                    st.metric("Probability", f"{prediction:.4f}", f"{prediction * 100:.1f}%")
                    st.progress(prediction)
                    
                    # Provide interpretation
                    if prediction >= 0.8:
                        st.success("Very high chance of admission!")
                    elif prediction >= 0.6:
                        st.info("Good chance of admission. You have a competitive profile.")
                    elif prediction >= 0.4:
                        st.warning("Moderate chance. Consider strengthening your application.")
                    else:
                        st.error("Lower chance of admission. Consider additional preparation or different programs.")
                    
                    # Find recommended universities based on the selected rating
                    try:
                        # Calculate the difference between the selected rating and each university's rating
                        uni_data['rating_diff'] = abs(uni_data['rating'] - uni_rating)
                        
                        # Sort by the difference and get the top 5
                        recommended_unis = uni_data.sort_values(by='rating_diff').head(5)
                        
                        # Display the recommended universities
                        st.subheader("Top 5 Recommended Universities")
                        st.dataframe(recommended_unis[['name', 'rating', 'type', 'state']], hide_index=True, column_config={
                            "name": "University Name",
                            "rating": st.column_config.NumberColumn("Rating", format="%.1f"),
                            "type": "Type",
                            "state": "State"
                        })
                    except Exception as e:
                        st.error(f"Error recommending universities: {e}")
                    
                    # Generate improvement suggestions based on target university rank
                    if target_rank != uni_rating:
                        try:
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
                            
                            st.subheader("Improvement Suggestions")
                            for suggestion in suggestions:
                                st.markdown(suggestion)
                            
                            # Calculate improved prediction if pursuing higher-ranked university
                            if target_rank < uni_rating:
                                improved_data = input_data.copy()
                                improved_data['University Rating'] = target_rank
                                improved_df = pd.DataFrame([improved_data])
                                improved_prediction = model.predict(improved_df)[0]
                                improved_prediction = max(0, min(1, improved_prediction))
                                
                                st.subheader("Predicted Chance with Target Ranking")
                                st.metric("Probability with Improvements", 
                                        f"{improved_prediction:.4f}", 
                                        f"{(improved_prediction - prediction) * 100:.1f}%")
                                st.progress(improved_prediction)
                                
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

    with st.sidebar:
        st.title("About")
        st.markdown("""
        ## AdmissionPulse

        This application helps students with:

        - Creating professional Statements of Purpose
        - Rating existing SOPs for quality
        - Generating Letters of Recommendation
        - Predicting admission chances
        - Getting personalized improvement suggestions

        Use the tabs above to navigate through different features.
        """)
        st.divider()
        st.markdown("""
        ### How to use
        1. Select a tab for the function you need
        2. Fill in the required information
        3. Submit the form to get results
        4. Download generated documents if needed
        5. For admission prediction, specify a target rank to get improvement suggestions
        """)

def login_page():
    st.title("AdmissionPulse 🎓")
    st.subheader("Login to Your Account")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("Sign Up"):
        st.session_state['show_signup'] = True
        st.rerun()

def signup_page():
    st.title("AdmissionPulse 🎓")
    st.subheader("Create a New Account")

    new_username = st.text_input("Choose a Username")
    new_email = st.text_input("Email Address")
    full_name = st.text_input("Full Name")
    new_password = st.text_input("Create Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Create Account"):
        if not new_username or not new_email or not full_name or not new_password:
            st.error("Please fill in all fields")
        elif new_password != confirm_password:
            st.error("Passwords do not match")
        else:
            success = create_user(new_username, new_password, new_email, full_name)
            if success:
                st.success("Account created successfully!")
                st.info("You can now log in.")
                st.session_state['show_signup'] = False
                st.rerun()
            else:
                st.error("Username or email already exists")

    if st.button("Back to Login"):
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
        st.sidebar.title(f"Welcome, {st.session_state.get('username', 'User')}")
        if st.sidebar.button("Logout"):
            logout()
            st.rerun()
        main_app()

if __name__ == "__main__":
    app()
