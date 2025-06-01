from datasets import Dataset
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

# Tokenize data and save it to prevent needing to retokenize each time
def tokenize_and_cache(
    emails,
    tokenizer_name="bert-base-uncased",
    max_length=320,
    save_path="tokenized_emails.pt",
    force_retokenize=False,
):
    # If file exists and we don't want to re-tokenize
    if Path(save_path).exists() and not force_retokenize:
        print(f"Loading tokenized data from '{save_path}'...")
        return torch.load(save_path)

    print(f"Tokenizing {len(emails)} emails with max_length={max_length}...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenized = tokenizer(
        emails,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    torch.save(tokenized, save_path)
    print(f"Saved tokenized data to '{save_path}'")

    return tokenized

# Helper function that saves the labels associated with the test/training set
def save_labels(labels, path):
    torch.save(torch.tensor(labels), path)
    print(f"[INFO] Saved labels to '{path}'")

# Helper function to compute performance metrics for the model
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = logits[:, 1]  # Probability/logit for the positive class (class 1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        # This happens if all labels are 0 or 1 in eval set
        roc_auc = float('nan')

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
    }

if __name__ == "__main__":
    # Load in the dataset
    df = pd.read_csv("clean_data_no_stop.csv")
    df["cleaned text"] = df["cleaned text"].astype(str) # ensure all text is saved as a string

    emails = df['cleaned text'].tolist()
    labels = df['label'].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(emails, labels, test_size=0.3, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"[INFO] Dataset sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create output directory if needed
    report_dir = "DBert_results"
    os.makedirs(report_dir, exist_ok=True)

    '''
    # Determine what the maximum number of tokens the tokenizer should use

    lengths = [len(tokenizer.tokenize(str(text))) for text in emails]

    print("Min:", np.min(lengths))
    print("Max:", np.max(lengths))
    print("Mean:", np.mean(lengths))
    print("Median:", np.median(lengths))
    print("99th percentile:", np.percentile(lengths, 99))

    # Outliers are present, plot the 99th percentile to get a better distribution of the data
    percentile_99 = np.percentile(lengths, 99)
    trimmed_lengths = [l for l in lengths if l <= percentile_99]

    plt.hist(trimmed_lengths, bins=50)
    plt.xlabel('Tokenized email length (trimmed at 99th percentile)')
    plt.ylabel('Number of emails')
    plt.show()
    '''
    ## BERT tokenization
    '''
    # Tokenize the data with a max token length of 320
    train_320 = tokenize_and_cache(emails=X_train, save_path='Bert_data\\train_320.pt')
    val_320 = tokenize_and_cache(emails=X_val, save_path='Bert_data\\val_320.pt')
    test_320 = tokenize_and_cache(emails=X_test, save_path='Bert_data\\test_320.pt')

    # Tokenize the data with a max token length of 512 to determine if more context is more beneficial to the model
    train_512 = tokenize_and_cache(emails=X_train, max_length=512, save_path='Bert_data\\train_512.pt')
    val_512 = tokenize_and_cache(emails=X_val, max_length=512, save_path='Bert_data\\val_512.pt')
    test_512 = tokenize_and_cache(emails=X_test, max_length=512, save_path='Bert_data\\test_512.pt')

    # Save the labels for each group
    save_labels(y_train, "Bert_data\\train_labels.pt")
    save_labels(y_val, "Bert_data\\val_labels.pt")
    save_labels(y_test, "Bert_data\\test_labels.pt")
    '''
    ## DistiliBERT tokenization
    
     # Tokenize the data with a max token length of 320
    train_320 = tokenize_and_cache(emails=X_train, tokenizer_name="distilbert-base-uncased", save_path='DBert_data\\train_320.pt')
    val_320 = tokenize_and_cache(emails=X_val, tokenizer_name="distilbert-base-uncased", save_path='DBert_data\\val_320.pt')
    test_320 = tokenize_and_cache(emails=X_test, tokenizer_name="distilbert-base-uncased", save_path='DBert_data\\test_320.pt')

    '''
    # Tokenize the data with a max token length of 512 to determine if more context is more beneficial to the model
    train_512 = tokenize_and_cache(emails=X_train, tokenizer_name="distilbert-base-uncased", max_length=512, save_path='DBert_data\\train_512.pt')
    val_512 = tokenize_and_cache(emails=X_val, tokenizer_name="distilbert-base-uncased", max_length=512, save_path='DBert_data\\val_512.pt')
    test_512 = tokenize_and_cache(emails=X_test, tokenizer_name="distilbert-base-uncased", max_length=512, save_path='DBert_data\\test_512.pt')

    
    # Save the labels for each group
    save_labels(y_train, "DBert_data\\train_labels.pt")
    save_labels(y_val, "DBert_data\\val_labels.pt")
    save_labels(y_test, "DBert_data\\test_labels.pt")
    '''

    # DistilBert Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2  # Binary classification
    )

    # Convert tokenized data into a dataset for the model
    train_dataset = Dataset.from_dict(train_320)
    train_dataset = train_dataset.add_column("labels", y_train)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset = Dataset.from_dict(val_320)
    val_dataset = val_dataset.add_column("labels", y_val)
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    training_args = TrainingArguments(
        output_dir="DBert_data\\results2",    # Where to save model and checkpoints
        evaluation_strategy="epoch",          # Evaluate after each epoch
        save_strategy="epoch",                # Save a checkpoint after each epoch
        learning_rate=2e-5,                   # Initial learning rate for AdamW optimizer
        per_device_train_batch_size=16,       # Batch size for training (per GPU)
        per_device_eval_batch_size=64,        # Batch size for evaluation
        num_train_epochs=3,                   # Number of full passes through the training set
        weight_decay=0.01,                    # Regularization to avoid overfitting
        load_best_model_at_end=True,          # Load best model based on evaluation metric
        logging_dir="DBert_data\\logs2",      # Directory for logs
        logging_steps=10,                     # How often to log during training
        metric_for_best_model="f1",           # Use the F1 score calculated when computing metrics to determine the best performing model
        greater_is_better=True,               # Maximize F1 score
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Stop if the F1 score does not improve after 2 evaluation rounds (epochs)
    )

    print("Training the model")
    trainer.train()

    # Extract optimizer info
    optimizer_type = type(trainer.optimizer).__name__
    optimizer_params = trainer.optimizer.param_groups[0]  # Get the first param group

    learning_rate = optimizer_params.get("lr", "N/A")
    weight_decay = training_args.weight_decay
    betas = optimizer_params.get("betas", ("N/A", "N/A"))
    epsilon = training_args.adam_epsilon if hasattr(training_args, "adam_epsilon") else "N/A"

    # DistilBERT uses CrossEntropyLoss for classification by default
    loss_type = "CrossEntropyLoss"

    # Extract metrics from training logs
    epochs = []
    train_loss = []
    eval_loss = []
    accuracy = []
    precision = []
    recall = []
    f1_scores = []

    for entry in trainer.state.log_history:
        if "epoch" in entry:
            epoch = entry["epoch"]
            if "loss" in entry:
                train_loss.append(entry["loss"])
            if "eval_loss" in entry:
                eval_loss.append(entry["eval_loss"])
            if "eval_accuracy" in entry:
                accuracy.append(entry["eval_accuracy"])
            if "eval_precision" in entry:
                precision.append(entry["eval_precision"])
            if "eval_recall" in entry:
                recall.append(entry["eval_recall"])
            if "eval_f1" in entry:
                f1_scores.append(entry["eval_f1"])
            epochs.append(epoch)

    # Ensure unique epoch list
    unique_epochs = sorted(set(epochs))
    train_loss = train_loss[:len(unique_epochs)]
    eval_loss = eval_loss[:len(unique_epochs)]
    accuracy = accuracy[:len(unique_epochs)]
    precision = precision[:len(unique_epochs)]
    recall = recall[:len(unique_epochs)]
    f1_scores = f1_scores[:len(unique_epochs)]

    # Find best epoch by F1
    best_f1 = max(f1_scores)
    best_epoch_index = f1_scores.index(best_f1)
    best_epoch = unique_epochs[best_epoch_index]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot training & validation loss
    ax1.plot(unique_epochs, train_loss, label="Training Loss", linestyle='--', marker='o', color='gray')
    ax1.plot(unique_epochs, eval_loss, label="Validation Loss", linestyle='--', marker='x', color='black')
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss Over Epochs")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # Plot classification metrics
    ax2.plot(unique_epochs, accuracy, label="Accuracy", marker='o', color='blue')
    ax2.plot(unique_epochs, precision, label="Precision", marker='^', color='orange')
    ax2.plot(unique_epochs, recall, label="Recall", marker='v', color='purple')
    ax2.plot(unique_epochs, f1_scores, label="F1 Score", marker='s', color='green')

    # Annotate best F1 score
    ax2.scatter(best_epoch, best_f1, color='red', s=100, zorder=5)
    ax2.annotate(f"Best F1: {best_f1:.4f}\nEpoch {best_epoch}",
                (best_epoch, best_f1),
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='red')

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Evaluation Metrics Over Epochs")
    ax2.legend(loc="lower right")
    ax2.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    loss_plot_path = os.path.join(report_dir, "loss_and_metrics_over_time.png")
    plt.savefig(loss_plot_path)
    plt.close()

    print("Evalutating the model using the test set")
    # Evalute using the test dataset
    test_dataset = Dataset.from_dict(test_320)
    test_dataset = test_dataset.add_column("labels", y_test)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    eval_results = trainer.evaluate(test_dataset)

    # Save results to file
    with open(os.path.join(report_dir, "test_eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    print("Saved evaluation results:", eval_results)

    # Get model predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Phish"])
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix - Phishing Detection")
    plt.tight_layout()
    cm_path1 = os.path.join(report_dir, "confusion_matrix.png")
    plt.savefig(cm_path1)
    plt.close()

    # Confusion matrix (normalized)
    cm = confusion_matrix(y_true, y_pred, normalize='true')  # normalize by true labels (rows)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Phish"])
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Normalized Confusion Matrix - Phishing Detection")
    plt.tight_layout()
    cm_path2 = os.path.join(report_dir, "confusion_matrix_normalized.png")
    plt.savefig(cm_path2)
    plt.close()

    print("Saving the model")
    # Save the model
    trainer.save_model("phishing-distilbert-model2")

    # Classification report
    print("Creating classification report")
    class_report = classification_report(y_true, y_pred, target_names=["Legit", "Phish"], digits=4)

    # Full report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_text = f"""
    ==================== EVALUATION REPORT ====================

    Timestamp       : {timestamp}
    Model           : distilbert-base-uncased
    Tokenizer       : distilbert-base-uncased
    Token Length    : 320
    Max Epochs      : {training_args.num_train_epochs}
    Best Metric     : {training_args.metric_for_best_model}

    ---------------- Optimizer & Loss Info ----------------
    Optimizer       : {optimizer_type}
    Learning Rate   : {learning_rate}
    Weight Decay    : {weight_decay}
    Betas           : {betas}
    Epsilon         : {epsilon}
    Loss Function   : {loss_type}

    ---------------- Evaluation Metrics ----------------
    {json.dumps(eval_results, indent=4)}

    Metrics Plot Saved at: {loss_plot_path}

    ---------------- Classification Report ----------------
    {class_report}

    Confusion Matrix Saved at: {cm_path1}
    Normalized Confusion Matrix Saved at: {cm_path2}

    ============================================================
    """

    # Save to text file
    report_file = os.path.join(report_dir, "final_evaluation_report.txt")
    with open(report_file, "w") as f:
        f.write(report_text.strip())

    print(f"[INFO] Final evaluation report saved to: {report_file}")