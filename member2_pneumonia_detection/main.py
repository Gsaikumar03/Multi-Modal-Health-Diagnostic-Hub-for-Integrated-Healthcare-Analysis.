from src.dataset_explorer import count_images, print_dataset_stats
from src.image_loader import load_random_images, check_image_properties
import os
import tensorflow as tf
from src.tf_data_pipeline import create_datasets, apply_normalization
from src.cnn_model import build_cnn
from src.transfer_model import build_transfer_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.advanced_metrics import evaluate_model_met
import numpy as np

DATA_PATH = "data/chest_xray"
MODEL_DIR = "/kaggle/working/saved_models"

os.makedirs(MODEL_DIR, exist_ok=True)


def print_separator(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


def main():

    # =====================================================
    # STEP 1: DATA LOADING
    # =====================================================

    print_separator("STEP 1: Loading Dataset")

    stats = count_images(DATA_PATH)
    print_dataset_stats(stats)

    load_random_images(DATA_PATH, split="train", label="NORMAL")
    load_random_images(DATA_PATH, split="train", label="PNEUMONIA")

    check_image_properties(DATA_PATH, split="train", label="NORMAL")

    train_ds, val_ds, test_ds = create_datasets(DATA_PATH)
    train_ds, val_ds, test_ds = apply_normalization(train_ds, val_ds, test_ds)

    print("Datasets loaded successfully.\n")


    # =====================================================
    # 2️⃣ BASELINE CNN MODEL
    # =====================================================

    print_separator("STEP 2: Training Baseline CNN")

    cnn_model = build_cnn()

    cnn_history = train_model(
        model=cnn_model,
        train_ds=train_ds,
        val_ds=val_ds
    )

    print_separator("Evaluating Baseline CNN")

    # Get predictions
    cnn_y_true, cnn_y_prob = evaluate_model(cnn_model, test_ds)

    # Compute metrics
    cnn_metrics = evaluate_model_met(cnn_y_true, cnn_y_prob)

    # Save model
    cnn_model_path = os.path.join(MODEL_DIR, "cnn_baseline.h5")
    cnn_model.save(cnn_model_path)
    print(f"Baseline CNN saved at: {cnn_model_path}")


    # =====================================================
    # 3️⃣ TRANSFER LEARNING (densenet)
    # =====================================================

    print_separator("STEP 3: Training Transfer Learning Model (DenseNet)")

    transfer_model = build_transfer_model()

    transfer_history = train_model(
        model=transfer_model,
        train_ds=train_ds,
        val_ds=val_ds
    )

    print_separator("Evaluating Transfer Learning Model")

    transfer_y_true, transfer_y_prob = evaluate_model(transfer_model, test_ds)
    transfer_metrics = evaluate_model_met(transfer_y_true, transfer_y_prob)

    transfer_model_path = os.path.join(MODEL_DIR, "densenet_pneumonia.h5")
    transfer_model.save(transfer_model_path)
    print(f"Transfer model saved at: {transfer_model_path}")


    # =====================================================
    # 4️⃣ FINAL COMPARISON
    # =====================================================

    print_separator("FINAL COMPARISON")

    print("Baseline CNN Metrics:")
    for key, value in cnn_metrics.items():
        print(f"{key:15s}: {value:.4f}")

    print("\nTransfer Learning Metrics - DenseNet:")
    for key, value in transfer_metrics.items():
        print(f"{key:15s}: {value:.4f}")

    print("\nTraining & Evaluation Completed Successfully 🚀")


if __name__ == "__main__":
    main()