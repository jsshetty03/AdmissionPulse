import sqlite3

DATABASE_FILE = "users.db"

def create_users_table():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            program TEXT,
            university TEXT,
            field_interest TEXT,
            career_goal TEXT,
            subjects_studied TEXT,
            projects_internships TEXT,
            lacking_skills TEXT,
            program_benefits TEXT,
            contribution TEXT,
            sop TEXT,
            username TEXT UNIQUE,
            password TEXT,
            email TEXT UNIQUE,
            full_name TEXT,
            student_name TEXT,
            relationship TEXT,
            subjects_taught TEXT,
            academic_performance TEXT,
            projects_completed TEXT,
            lor_content TEXT
            gre_score INTEGER,
            toefl_score INTEGER,
            university_reting INTEGER,
            sop_strength REAL,
            cgpa REAL,
            research INTEGER
        )
    """)

    conn.commit()
    conn.close()

def save_user_data(responses, sop, username=None):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Check if this is for an existing user
    if username:
        # Try to update existing user's SOP info
        cursor.execute("""
            UPDATE users SET
                program = ?, university = ?, field_interest = ?, career_goal = ?,
                subjects_studied = ?, projects_internships = ?, lacking_skills = ?,
                program_benefits = ?, contribution = ?, sop = ?
            WHERE username = ?
        """, (
            responses['program'], responses['university'], responses['field_interest'], responses['career_goal'],
            responses['subjects_studied'], responses['projects_internships'], responses['lacking_skills'],
            responses['program_benefits'], responses['contribution'], sop, username
        ))
        
        # If no rows were updated, insert a new record
        if cursor.rowcount == 0:
            cursor.execute("""
                INSERT INTO users (
                    program, university, field_interest, career_goal,
                    subjects_studied, projects_internships, lacking_skills,
                    program_benefits, contribution, sop, username
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                responses['program'], responses['university'], responses['field_interest'], responses['career_goal'],
                responses['subjects_studied'], responses['projects_internships'], responses['lacking_skills'],
                responses['program_benefits'], responses['contribution'], sop, username
            ))
    else:
        # Insert new record without username
        cursor.execute("""
            INSERT INTO users (
                program, university, field_interest, career_goal,
                subjects_studied, projects_internships, lacking_skills,
                program_benefits, contribution, sop
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            responses['program'], responses['university'], responses['field_interest'], responses['career_goal'],
            responses['subjects_studied'], responses['projects_internships'], responses['lacking_skills'],
            responses['program_benefits'], responses['contribution'], sop
        ))

    conn.commit()
    conn.close()

def save_lor_data(lor_data, lor_content, username=None):
    """
    Save letter of recommendation data to the users table
    
    Args:
        lor_data (dict): Dictionary containing LOR details
        lor_content (str): The generated letter of recommendation text
        username (str, optional): Username to associate with this LOR
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    if username:
        # Update existing user's record with LOR data
        cursor.execute("""
            UPDATE users SET
                student_name = ?,
                relationship = ?,
                subjects_taught = ?,
                academic_performance = ?,
                projects_completed = ?,
                lor_content = ?
            WHERE username = ?
        """, (
            lor_data['for_whom'],
            lor_data['how_you_know'],
            lor_data['subjects_taught'],
            lor_data['marks'],
            lor_data['projects'],
            lor_content,
            username
        ))
        
        # If no rows were updated, insert a new record
        if cursor.rowcount == 0:
            cursor.execute("""
                INSERT INTO users (
                    student_name, relationship, subjects_taught,
                    academic_performance, projects_completed, lor_content, username
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                lor_data['for_whom'],
                lor_data['how_you_know'],
                lor_data['subjects_taught'],
                lor_data['marks'],
                lor_data['projects'],
                lor_content,
                username
            ))
    else:
        # Insert new record with just LOR data
        cursor.execute("""
            INSERT INTO users (
                student_name, relationship, subjects_taught,
                academic_performance, projects_completed, lor_content
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            lor_data['for_whom'],
            lor_data['how_you_know'],
            lor_data['subjects_taught'],
            lor_data['marks'],
            lor_data['projects'],
            lor_content
        ))
    
    conn.commit()
    conn.close()

def save_admission_prediction(admission_data, username=None):
    """
    Save admission prediction data to the database
    
    Args:
        admission_data (dict): Dictionary containing prediction details (GRE, TOEFL, ratings, etc.)
        username (str, optional): Username to associate this prediction with
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    if username:
        # Update existing user record
        cursor.execute("""
            UPDATE users SET
                gre_score = ?,
                toefl_score = ?,
                university_reting = ?,
                sop_strength = ?,
                cgpa = ?,
                research = ?
            WHERE username = ?
        """, (
            admission_data['GRE Score'],
            admission_data['TOEFL Score'], 
            admission_data['University Rating'],
            admission_data['SOP'],
            admission_data['CGPA'],
            admission_data['Research'],
            username
        ))
        
        # If no rows updated, insert new record with username
        if cursor.rowcount == 0:
            cursor.execute("""
                INSERT INTO users (
                    gre_score, toefl_score, university_reting,
                    sop_strength, cgpa, research, username
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                admission_data['GRE Score'],
                admission_data['TOEFL Score'],
                admission_data['University Rating'],
                admission_data['SOP'],
                admission_data['CGPA'],
                admission_data['Research'],
                username
            ))
    else:
        # Insert new record without username
        cursor.execute("""
            INSERT INTO users (
                gre_score, toefl_score, university_reting,
                sop_strength, cgpa, research
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            admission_data['GRE Score'],
            admission_data['TOEFL Score'],
            admission_data['University Rating'],
            admission_data['SOP'],
            admission_data['CGPA'],
            admission_data['Research']
        ))
    
    conn.commit()
    conn.close()
    
    return True

# Ensure the table is created
create_users_table()
