from flask import Flask, request, render_template
from improved_sop_rater import ImprovedSOPRater
from sop_generator import generate_sop
from lor_generator import generate_lor  # Import the LOR generator
from database import save_user_data

app = Flask(__name__)

# Initialize the rater at startup
rater = ImprovedSOPRater()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        responses = {
            "program": request.form["program"],
            "university": request.form["university"],
            "field_interest": request.form["field_interest"],
            "career_goal": request.form["career_goal"],
            "subjects_studied": request.form["subjects_studied"],
            "projects_internships": request.form["projects_internships"],
            "lacking_skills": request.form["lacking_skills"],
            "program_benefits": request.form["program_benefits"],
            "contribution": request.form["contribution"]
        }
        
        # Generate SOP
        sop = generate_sop(responses)
        
        # Rate SOP
        result = rater.rate_sop(sop)
        
        # Save to database
        save_user_data(responses, sop)
        
        return render_template(
            "result.html", 
            sop=sop, 
            rating=result['rating'],
            confidence=result['confidence']
        )
    
    return render_template("index.html")

@app.route("/rate_prewritten_sop", methods=["GET", "POST"])
def rate_prewritten_sop():
    if request.method == "POST":
        # Get the prewritten SOP from the form
        prewritten_sop = request.form["prewritten_sop"]
        
        # Rate the prewritten SOP
        result = rater.rate_sop(prewritten_sop)
        
        # Display the rating and confidence score
        return render_template(
            "prewritten_result.html", 
            sop=prewritten_sop, 
            rating=result['rating'],
            confidence=result['confidence']
        )
    
    return render_template("rate_prewritten.html")

# New route for LOR generator
@app.route("/generate_lor", methods=["GET", "POST"])
def generate_lor_route():
    if request.method == "POST":
        responses = {
            "for_whom": request.form["for_whom"],
            "how_you_know": request.form["how_you_know"],
            "subjects_taught": request.form["subjects_taught"],
            "marks": request.form["marks"],
            "projects": request.form["projects"]
        }
        
        # Generate LOR
        lor = generate_lor(responses)
        
        return render_template("lor_result.html", lor=lor)
    
    return render_template("generate_lor.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)