import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./sop_bert_model"  # Path to the fine-tuned model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

def rate_sop(sop_text):
    # Tokenize the input SOP text
    inputs = tokenizer(sop_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to a rating (1-5)
    logits = outputs.logits
    score = torch.argmax(logits, dim=1).item() + 1  # Convert to scale 1-5
    
    return score

# Example usage
if __name__ == "__main__":
    sop_text = "I have always been passionate about Computer Science. My goal is to specialize in Artificial Intelligence and contribute to developing autonomous systems."
    rating = rate_sop(sop_text)
    print(f"Predicted Rating: {rating}")