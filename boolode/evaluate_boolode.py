from pathlib import Path

import numpy as np
import pandas as pd


NUM_GENES = 18

RESULT_DIR = Path("logs_boolode/dyn-LL-1")
REF_NETWORK = Path(
    "boolode_data/raw/dyn-LL-1/refNetwork.csv"
)


def create_edge_list(num_genes):
    edge_list = []

    for receiver in range(num_genes):
        for sender in range(num_genes):
            if receiver != sender:
                edge_list.append((receiver, sender))

    return edge_list


def main():

    predicted_path = (
        RESULT_DIR / "test_predicted_edges.npy"
    )

    predicted_edges = np.load(predicted_path)

    gene_names_path = Path(
        "boolode_data/processed/dyn-LL-1/gene_names.txt"
    )

    with open(gene_names_path, encoding="utf-8") as f:
        gene_names = [
            line.strip()
            for line in f
            if line.strip()
        ]

    gene_to_index = {
        name: i
        for i, name in enumerate(gene_names)
    }

    edge_list = create_edge_list(NUM_GENES)

    edge_to_index = {
        edge: i
        for i, edge in enumerate(edge_list)
    }

    true_edges = np.zeros(
        len(edge_list),
        dtype=np.int64
    )

    ref_network = pd.read_csv(REF_NETWORK)

    for _, row in ref_network.iterrows():

        receiver_name = row["Gene2"]
        sender_name = row["Gene1"]

        if (
            receiver_name not in gene_to_index
            or sender_name not in gene_to_index
        ):
            continue

        receiver = gene_to_index[receiver_name]
        sender = gene_to_index[sender_name]

        if receiver == sender:
            continue

        edge_index = edge_to_index[
            (receiver, sender)
        ]

        true_edges[edge_index] = 1

    # テストデータ45個の多数決
    type1_ratio = np.mean(
        predicted_edges == 1,
        axis=0
    )

    prediction_a = (
        type1_ratio >= 0.5
    ).astype(np.int64)

    prediction_b = 1 - prediction_a

    print("===== データ確認 =====")
    print("予測形状:", predicted_edges.shape)
    print("正解エッジ数:", np.sum(true_edges))
    print("全エッジ候補数:", len(true_edges))

    for name, prediction in [
        ("A: タイプ1 = 関係あり", prediction_a),
        ("B: タイプ0 = 関係あり", prediction_b),
    ]:

        tp = np.sum(
            (true_edges == 1)
            & (prediction == 1)
        )

        tn = np.sum(
            (true_edges == 0)
            & (prediction == 0)
        )

        fp = np.sum(
            (true_edges == 0)
            & (prediction == 1)
        )

        fn = np.sum(
            (true_edges == 1)
            & (prediction == 0)
        )

        accuracy = (tp + tn) / len(true_edges)

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        print("")
        print("=====", name, "=====")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("TP:", tp)
        print("TN:", tn)
        print("FP:", fp)
        print("FN:", fn)


if __name__ == "__main__":
    main()
