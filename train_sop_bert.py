if __name__ == '__main__':
    import torch
    import pandas as pd
    import numpy as np
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight

    # Load Dataset
    df = pd.read_csv("sop_dataset.csv")

    # Train-Validation Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["sop_text"].tolist(),
        df["rating"].tolist(),
        test_size=0.2,
        stratify=df["rating"].tolist() if len(df) >= 10 else None
    )

    # Tokenizer
    MODEL_NAME = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['rating']),
        y=df['rating']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Custom Trainer with class weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):# ✅ Add **kwargs to accept extra arguments
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Load Pretrained BERT Model
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
    model.to(device)

    # Training Arguments (Fix multiprocessing issue)
    training_args = TrainingArguments(
        output_dir="./sop_bert_model",
        eval_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=False if torch.backends.mps.is_available() else True,
        dataloader_num_workers=0,  # 🚀 **Fixes Mac multiprocessing issue**
    )

    # Compute Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": accuracy, "f1": f1}

    # Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train Model
    trainer.train()

    # Save Model
    model.save_pretrained("./sop_bert_model")
    tokenizer.save_pretrained("./sop_bert_model")
