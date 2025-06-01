from datasets import Dataset
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed

def tokenize_and_cache(
    emails,
    tokenizer_name="distilbert-base-uncased",
    max_length=320,
    save_path="tokenized_emails.pt",
    force_retokenize=False,
):
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

def save_classification_report(report_str, filename):
    with open(filename, 'w') as f:
        f.write(report_str)
    print(f"Saved classification report to {filename}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = logits[:, 1] if logits.shape[1] > 1 else logits[:, 0]

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float('nan')

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
    }

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("clean_data_no_stop.csv")
    df["cleaned text"] = df["cleaned text"].astype(str)  # ensure all text is string
    emails = df['cleaned text'].tolist()
    labels = df['label'].tolist()

    # Split off a separate test set (e.g., 15% for testing)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        emails, labels, test_size=0.15, stratify=labels, random_state=42
    )

    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    data_dir = "DBert_data"
    report_dir = "DBert_results"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    all_fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val), 1):
        print(f"\n===== Fold {fold}/{k} =====")

        X_train_fold = [X_train_val[i] for i in train_idx]
        y_train_fold = [y_train_val[i] for i in train_idx]
        X_val_fold = [X_train_val[i] for i in val_idx]
        y_val_fold = [y_train_val[i] for i in val_idx]

        # Tokenize fold data
        train_tokens = tokenize_and_cache(
            X_train_fold,
            tokenizer_name="distilbert-base-uncased",
            save_path=os.path.join(data_dir, f"train_fold{fold}.pt"),
            force_retokenize=True,
        )
        val_tokens = tokenize_and_cache(
            X_val_fold,
            tokenizer_name="distilbert-base-uncased",
            save_path=os.path.join(data_dir, f"val_fold{fold}.pt"),
            force_retokenize=True,
        )

        # Create datasets with labels
        train_dataset = Dataset.from_dict(train_tokens).add_column("labels", y_train_fold)
        val_dataset = Dataset.from_dict(val_tokens).add_column("labels", y_val_fold)
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        set_seed(42)
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

        fold_output_dir = os.path.join(report_dir, f"fold{fold}_results")

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=os.path.join(data_dir, "logs"),
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print(f"Training Model on fold {fold}")
        trainer.train()

        print(f"Obtaining fold metrics for fold {fold}")
        fold_metrics_over_epochs = []
        for entry in trainer.state.log_history:
            if "eval_f1" in entry and "epoch" in entry:
                fold_metrics_over_epochs.append({
                    "fold": fold,
                    "epoch": entry["epoch"],
                    "f1": entry["eval_f1"],
                    "accuracy": entry.get("eval_accuracy"),
                    "precision": entry.get("eval_precision"),
                    "recall": entry.get("eval_recall"),
                })
        all_fold_metrics.extend(fold_metrics_over_epochs)

        # Classification report & confusion matrix for validation set
        val_preds = trainer.predict(val_dataset)
        y_val_pred = val_preds.predictions.argmax(axis=1)
        y_val_true = val_preds.label_ids

        val_class_report = classification_report(
            y_val_true, y_val_pred, target_names=["Legit", "Phish"], digits=4
        )
        print(f"\nClassification report for Fold {fold} Validation:\n{val_class_report}")

        report_path = os.path.join(report_dir, f"fold{fold}_val_classification_report.txt")
        save_classification_report(val_class_report, report_path)

        cm_val = confusion_matrix(y_val_true, y_val_pred)
        disp_val = ConfusionMatrixDisplay(cm_val, display_labels=["Legit", "Phish"])
        disp_val.plot(cmap='Blues')
        plt.title(f"Confusion Matrix - Validation Fold {fold}")
        plt.savefig(os.path.join(report_dir, f"fold{fold}_val_confusion_matrix.png"))
        plt.close()

    # Aggregate and save fold metrics over time
    metrics_df = pd.DataFrame(all_fold_metrics)
    fold_metrics_path = os.path.join(report_dir, "fold_metrics_over_time.csv")
    metrics_df.to_csv(fold_metrics_path, index=False)
    print(f"Fold metrics over time saved to {fold_metrics_path}")

    # Plot F1 and other metrics over epochs per fold
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=metrics_df, x="epoch", y="f1", hue="fold", marker="o")
    plt.title("F1 Score over Epochs per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend(title="Fold")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "f1_over_time_across_folds.png"))
    plt.close()

    for metric in ["accuracy", "precision", "recall"]:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=metrics_df, x="epoch", y=metric, hue="fold", marker="o")
        plt.title(f"{metric.capitalize()} over Epochs per Fold")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.legend(title="Fold")
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, f"{metric}_over_time_across_folds.png"))
        plt.close()

    best_row = metrics_df.loc[metrics_df['f1'].idxmax()]
    best_fold = int(best_row['fold'])
    best_epoch = int(best_row['epoch'])

    print(f"\nBest model: Fold {best_fold} at epoch {best_epoch} with F1={best_row['f1']:.4f}")

    best_checkpoint_path = trainer.state.best_model_checkpoint
    print(f"Loading best checkpoint from: {best_checkpoint_path}")
    best_model = DistilBertForSequenceClassification.from_pretrained(best_checkpoint_path)

    # Prepare test dataset
    test_tokens = tokenize_and_cache(
        X_test,
        tokenizer_name="distilbert-base-uncased",
        save_path=os.path.join(data_dir, "test_set.pt"),
        force_retokenize=True,
    )
    test_dataset = Dataset.from_dict(test_tokens).add_column("labels", y_test)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Evaluate best model on test set
    test_training_args = TrainingArguments(
        output_dir=os.path.join(report_dir, "test_results"),
        per_device_eval_batch_size=64,
        do_train=False,
        do_eval=True,
    )
    test_trainer = Trainer(
        model=best_model,
        args=test_training_args,
        compute_metrics=compute_metrics,
    )

    # Extract optimizer info
    optimizer_type = type(test_trainer.optimizer).__name__
    optimizer_params = test_trainer.optimizer.param_groups[0]  # Get the first param group

    learning_rate = optimizer_params.get("lr", "N/A")
    weight_decay = test_training_args.weight_decay
    betas = optimizer_params.get("betas", ("N/A", "N/A"))
    epsilon = test_training_args.adam_epsilon if hasattr(test_training_args, "adam_epsilon") else "N/A"

    # DistilBERT uses CrossEntropyLoss for classification by default
    loss_type = "CrossEntropyLoss"

    test_metrics = test_trainer.evaluate(test_dataset)
    print("\nTest set evaluation metrics:")
    print(test_metrics)

    test_preds = test_trainer.predict(test_dataset)
    y_test_pred = test_preds.predictions.argmax(axis=1)
    y_test_true = test_preds.label_ids

    # ROC Curve Visualization
    probs = test_preds.predictions[:, 1]
    fpr, tpr, _ = roc_curve(y_test_true, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Set")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(report_dir, "roc_curve_test_set.png"))
    plt.close()

    print(f"ROC Curve graph saved at {os.path.join(report_dir, "roc_curve_test_set.png")}")

    # Confusion matrix
    cm = confusion_matrix(y_test_true, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Phish"])
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix - Phishing Detection")
    plt.tight_layout()
    cm_path1 = os.path.join(report_dir, "test_confusion_matrix.png")
    plt.savefig(cm_path1)
    plt.close()

    # Confusion matrix (normalized)
    cm = confusion_matrix(y_test_true, y_test_pred, normalize='true')  # normalize by true labels (rows)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Phish"])
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Normalized Confusion Matrix - Phishing Detection")
    plt.tight_layout()
    cm_path2 = os.path.join(report_dir, "test_confusion_matrix_normalized.png")
    plt.savefig(cm_path2)
    plt.close()

    test_class_report = classification_report(
        y_test_true, y_test_pred, target_names=["Legit", "Phish"], digits=4
    )

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
    {json.dumps(test_metrics, indent=4)}

    ROC Curve visualization saved at: {os.path.join(report_dir, "roc_curve_test_set.png")}

    ---------------- Classification Report ----------------
    {test_class_report}

    Confusion Matrix Saved at: {cm_path1}
    Normalized Confusion Matrix Saved at: {cm_path2}

    ============================================================
    """

    print("\nClassification report for Test Set:\n", test_class_report)
    test_report_path = os.path.join(report_dir, "test_set_classification_report.txt")
    save_classification_report(report_text, test_report_path)
