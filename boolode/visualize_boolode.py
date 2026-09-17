import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

RESULT_DIR = (
    BASE_DIR
    / "logs_boolode"
    / "dyn-LL-1"
    / "experiment_epoch10"
)

METRICS_FILE = RESULT_DIR / "metrics.csv"


def main():
    epochs = []
    train_loss = []
    valid_loss = []
    type1_accuracy = []
    type0_accuracy = []
    type1_precision = []
    type1_recall = []

    with open(METRICS_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            valid_loss.append(float(row["valid_loss"]))
            type1_accuracy.append(float(row["type1_accuracy"]))
            type0_accuracy.append(float(row["type0_accuracy"]))
            type1_precision.append(float(row["type1_precision"]))
            type1_recall.append(float(row["type1_recall"]))

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, valid_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, type1_accuracy, label="Type 1 Accuracy")
    plt.plot(epochs, type0_accuracy, label="Type 0 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "accuracy_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, type1_precision, label="Precision")
    plt.plot(epochs, type1_recall, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Precision and Recall per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "precision_recall_curve.png", dpi=300)
    plt.close()

    print("グラフを保存しました。")
    print(f"保存先: {RESULT_DIR}")


if __name__ == "__main__":
    main()