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
            sop TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def save_user_data(responses, sop):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

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

# Ensure the table is created
create_users_table()
