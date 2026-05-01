from src.data_loader import load_data, basic_info
from src.eda import target_distribution, numerical_summary, categorical_summary

from src.preprocessing import load_data, preprocess_data, split_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.advanced_metrics import compute_all_metrics
from src.plots import plot_confusion_matrix, plot_roc_curve, plot_pr_curve

from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_PATH = "data/heart.csv"
TARGET_COLUMN = "Heart Disease"

def main():
    df = load_data(DATA_PATH)

    print("\n========== BASIC DATA INFO ==========")
    basic_info(df)

    print("\n========== EDA ==========")
    numerical_summary(df)
    categorical_summary(df)
    target_distribution(df, TARGET_COLUMN)

    X, y, scaler, feature_names = preprocess_data(df, TARGET_COLUMN)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = build_model(X_train.shape[1])

    train_model(model, X_train, y_train, X_val, y_val)

    evaluate_model(model, X_test, y_test)

    model.save("heart_model.h5")
    
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    metrics, cm, fpr, tpr, roc_auc, precision_curve, recall_curve, pr_auc = \
        compute_all_metrics(y_test, y_pred, y_prob)

    print(metrics)

    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr, roc_auc)
    plot_pr_curve(recall_curve, precision_curve, pr_auc)


if __name__ == "__main__":
    main()