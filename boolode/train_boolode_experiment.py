from __future__ import print_function

import os
import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# ============================================================
# パス設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from boolode_utils import load_boolode_data
from modules import MLPEncoder, MLPDecoder
from utils import (
    gumbel_softmax,
    nll_gaussian,
    kl_categorical_uniform,
)


# ============================================================
# エッジ構造の作成
# ============================================================

def create_rel_rec_send(num_atoms):
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)

    receivers = np.where(off_diag)[0]
    senders = np.where(off_diag)[1]

    rel_rec = np.zeros([len(receivers), num_atoms])
    rel_send = np.zeros([len(senders), num_atoms])

    rel_rec[np.arange(len(receivers)), receivers] = 1
    rel_send[np.arange(len(senders)), senders] = 1

    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    return rel_rec, rel_send


# ============================================================
# 正解ネットワークの読み込み
# ============================================================

def load_reference_edges(ref_path, gene_names, num_atoms):
    gene_to_index = {
        gene_name: index
        for index, gene_name in enumerate(gene_names)
    }

    true_edges = np.zeros(num_atoms * (num_atoms - 1), dtype=np.int64)

    edge_index = 0

    for receiver in range(num_atoms):
        for sender in range(num_atoms):
            if receiver == sender:
                continue

            true_edges[edge_index] = 0
            edge_index += 1

    with open(ref_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            receiver_name = row["Gene2"]
            sender_name = row["Gene1"]

            if receiver_name == sender_name:
                continue

            if receiver_name not in gene_to_index:
                continue

            if sender_name not in gene_to_index:
                continue

            receiver = gene_to_index[receiver_name]
            sender = gene_to_index[sender_name]

            edge_index = 0

            for r in range(num_atoms):
                for s in range(num_atoms):
                    if r == s:
                        continue

                    if r == receiver and s == sender:
                        true_edges[edge_index] = 1

                    edge_index += 1

    return true_edges


# ============================================================
# 評価指標
# ============================================================

def calculate_metrics(true_edges, predicted_edges):
    true_edges = np.asarray(true_edges)
    predicted_edges = np.asarray(predicted_edges)

    tp = np.sum((true_edges == 1) & (predicted_edges == 1))
    tn = np.sum((true_edges == 0) & (predicted_edges == 0))
    fp = np.sum((true_edges == 0) & (predicted_edges == 1))
    fn = np.sum((true_edges == 1) & (predicted_edges == 0))

    accuracy = (tp + tn) / len(true_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ============================================================
# エッジ予測の取得
# ============================================================

def collect_edge_predictions(
    data_loader,
    encoder,
    rel_rec,
    rel_send,
    args,
):
    encoder.eval()

    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            data = batch[0]

            if args.cuda:
                data = data.cuda()

            logits = encoder(data, rel_rec, rel_send)

            predictions = torch.argmax(logits, dim=-1)

            all_predictions.append(
                predictions.cpu().numpy()
            )

    return np.concatenate(all_predictions, axis=0)


def evaluate_edges(
    data_loader,
    encoder,
    rel_rec,
    rel_send,
    true_edges,
    args,
):
    predicted_edges = collect_edge_predictions(
        data_loader,
        encoder,
        rel_rec,
        rel_send,
        args,
    )

    type1_ratio = np.mean(predicted_edges == 1, axis=0)

    prediction_type1_positive = (
        type1_ratio >= 0.5
    ).astype(np.int64)

    prediction_type0_positive = (
        1 - prediction_type1_positive
    )

    metrics_type1 = calculate_metrics(
        true_edges,
        prediction_type1_positive,
    )

    metrics_type0 = calculate_metrics(
        true_edges,
        prediction_type0_positive,
    )

    return metrics_type1, metrics_type0


# ============================================================
# 損失計算
# ============================================================

def calculate_loss(
    data,
    encoder,
    decoder,
    rel_rec,
    rel_send,
    args,
):
    logits = encoder(
        data,
        rel_rec,
        rel_send,
    )

    edges = gumbel_softmax(
        logits,
        tau=args.temp,
        hard=args.hard,
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    output = decoder(
        data,
        edges,
        rel_rec,
        rel_send,
        pred_steps=args.prediction_steps,
    )

    target = data[:, :, 1:, :]

    min_timesteps = min(
        output.size(2),
        target.size(2),
    )

    output = output[:, :, :min_timesteps, :]
    target = target[:, :, :min_timesteps, :]

    loss_nll = nll_gaussian(
        output,
        target,
        args.var,
    )

    loss_kl = kl_categorical_uniform(
        probabilities,
        args.num_atoms,
        args.edge_types,
    )

    loss = loss_nll + loss_kl

    return loss, loss_nll, loss_kl


# ============================================================
# 1epochの学習
# ============================================================

def train_one_epoch(
    train_loader,
    encoder,
    decoder,
    optimizer,
    rel_rec,
    rel_send,
    args,
):
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    num_batches = 0

    for batch in train_loader:
        data = batch[0]

        if args.cuda:
            data = data.cuda()

        optimizer.zero_grad()

        loss, loss_nll, loss_kl = calculate_loss(
            data,
            encoder,
            decoder,
            rel_rec,
            rel_send,
            args,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_nll += loss_nll.item()
        total_kl += loss_kl.item()

        num_batches += 1

    return (
        total_loss / num_batches,
        total_nll / num_batches,
        total_kl / num_batches,
    )


# ============================================================
# Validation
# ============================================================

def evaluate_loss(
    data_loader,
    encoder,
    decoder,
    rel_rec,
    rel_send,
    args,
):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            data = batch[0]

            if args.cuda:
                data = data.cuda()

            loss, loss_nll, loss_kl = calculate_loss(
                data,
                encoder,
                decoder,
                rel_rec,
                rel_send,
                args,
            )

            total_loss += loss.item()
            total_nll += loss_nll.item()
            total_kl += loss_kl.item()

            num_batches += 1

    return (
        total_loss / num_batches,
        total_nll / num_batches,
        total_kl / num_batches,
    )


# ============================================================
# テスト結果の保存
# ============================================================

def save_test_results(
    test_loader,
    encoder,
    rel_rec,
    rel_send,
    true_edges,
    save_dir,
    args,
):
    encoder.eval()

    all_logits = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            data = batch[0]

            if args.cuda:
                data = data.cuda()

            logits = encoder(
                data,
                rel_rec,
                rel_send,
            )

            predictions = torch.argmax(
                logits,
                dim=-1,
            )

            all_logits.append(
                logits.cpu().numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

    all_logits = np.concatenate(all_logits, axis=0)
    all_predictions = np.concatenate(
        all_predictions,
        axis=0,
    )

    np.save(
        save_dir / "test_logits.npy",
        all_logits,
    )

    np.save(
        save_dir / "test_predicted_edges.npy",
        all_predictions,
    )

    type1_ratio = np.mean(
        all_predictions == 1,
        axis=0,
    )

    prediction_type1 = (
        type1_ratio >= 0.5
    ).astype(np.int64)

    prediction_type0 = 1 - prediction_type1

    metrics_type1 = calculate_metrics(
        true_edges,
        prediction_type1,
    )

    metrics_type0 = calculate_metrics(
        true_edges,
        prediction_type0,
    )

    with open(
        save_dir / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "type1_positive": metrics_type1,
                "type0_positive": metrics_type0,
            },
            file,
            indent=4,
        )

    print("\n===== Test Evaluation =====")
    print("Type 1 = relation exists")
    print(metrics_type1)

    print("\nType 0 = relation exists")
    print(metrics_type0)


# ============================================================
# メイン処理
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--encoder-hidden", type=int, default=256)
    parser.add_argument("--decoder-hidden", type=int, default=256)

    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--num-atoms", type=int, default=18)
    parser.add_argument("--edge-types", type=int, default=2)

    parser.add_argument("--timesteps", type=int, default=899)
    parser.add_argument("--dims", type=int, default=1)
    parser.add_argument("--prediction-steps", type=int, default=1)

    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--var", type=float, default=5e-5)

    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--no-factor", action="store_true")
    parser.add_argument("--cuda", action="store_true")

    parser.add_argument(
        "--save-dir",
        type=str,
        default="logs_boolode/dyn-LL-1/experiment",
    )

    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません。")

    args.save_dir = Path(args.save_dir)

    if not args.save_dir.is_absolute():
        args.save_dir = BASE_DIR / args.save_dir

    args.save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_dir = (
        BASE_DIR
        / "boolode_data"
        / "processed"
        / "dyn-LL-1"
    )

    ref_path = (
        BASE_DIR
        / "boolode_data"
        / "raw"
        / "dyn-LL-1"
        / "refNetwork.csv"
    )

    train_loader, valid_loader, test_loader = (
        load_boolode_data(
            data_dir=str(data_dir),
            batch_size=args.batch_size,
        )
    )

    rel_rec, rel_send = create_rel_rec_send(
        args.num_atoms,
    )

    if args.cuda:
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()

    gene_names_path = data_dir / "gene_names.txt"

    with open(
        gene_names_path,
        "r",
        encoding="utf-8",
    ) as file:
        gene_names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    true_edges = load_reference_edges(
        ref_path,
        gene_names,
        args.num_atoms,
    )

    encoder = MLPEncoder(
        n_in=args.timesteps * args.dims,
        n_hid=args.encoder_hidden,
        n_out=args.edge_types,
        do_prob=0.0,
        factor=not args.no_factor,
    )

    decoder = MLPDecoder(
        n_in_node=args.dims,
        edge_types=args.edge_types,
        msg_hid=args.decoder_hidden,
        msg_out=args.decoder_hidden,
        n_hid=args.decoder_hidden,
        do_prob=0.0,
    )

    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    optimizer = optim.Adam(
        list(encoder.parameters())
        + list(decoder.parameters()),
        lr=args.lr,
    )

    best_valid_loss = float("inf")

    metrics_path = args.save_dir / "metrics.csv"

    fieldnames = [
        "epoch",
        "train_loss",
        "train_nll",
        "train_kl",
        "valid_loss",
        "valid_nll",
        "valid_kl",
        "type1_accuracy",
        "type1_precision",
        "type1_recall",
        "type0_accuracy",
        "type0_precision",
        "type0_recall",
    ]

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_nll, train_kl = (
                train_one_epoch(
                    train_loader,
                    encoder,
                    decoder,
                    optimizer,
                    rel_rec,
                    rel_send,
                    args,
                )
            )

            valid_loss, valid_nll, valid_kl = (
                evaluate_loss(
                    valid_loader,
                    encoder,
                    decoder,
                    rel_rec,
                    rel_send,
                    args,
                )
            )

            metrics_type1, metrics_type0 = (
                evaluate_edges(
                    valid_loader,
                    encoder,
                    rel_rec,
                    rel_send,
                    true_edges,
                    args,
                )
            )

            print(
                f"\nEpoch {epoch:04d}"
            )

            print(
                f"Train Loss: {train_loss:.6f} "
                f"NLL: {train_nll:.6f} "
                f"KL: {train_kl:.6f}"
            )

            print(
                f"Valid Loss: {valid_loss:.6f} "
                f"NLL: {valid_nll:.6f} "
                f"KL: {valid_kl:.6f}"
            )

            print(
                "Type1 positive: "
                f"Accuracy={metrics_type1['accuracy']:.4f}, "
                f"Precision={metrics_type1['precision']:.4f}, "
                f"Recall={metrics_type1['recall']:.4f}"
            )

            print(
                "Type0 positive: "
                f"Accuracy={metrics_type0['accuracy']:.4f}, "
                f"Precision={metrics_type0['precision']:.4f}, "
                f"Recall={metrics_type0['recall']:.4f}"
            )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_nll": train_nll,
                    "train_kl": train_kl,
                    "valid_loss": valid_loss,
                    "valid_nll": valid_nll,
                    "valid_kl": valid_kl,
                    "type1_accuracy": metrics_type1["accuracy"],
                    "type1_precision": metrics_type1["precision"],
                    "type1_recall": metrics_type1["recall"],
                    "type0_accuracy": metrics_type0["accuracy"],
                    "type0_precision": metrics_type0["precision"],
                    "type0_recall": metrics_type0["recall"],
                }
            )

            file.flush()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                torch.save(
                    encoder.state_dict(),
                    args.save_dir / "encoder_best.pt",
                )

                torch.save(
                    decoder.state_dict(),
                    args.save_dir / "decoder_best.pt",
                )

                print("Best model saved.")

    with open(
        args.save_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vars(args),
            file,
            indent=4,
            default=str,
        )

    encoder.load_state_dict(
        torch.load(
            args.save_dir / "encoder_best.pt",
            map_location="cuda" if args.cuda else "cpu",
        )
    )

    save_test_results(
        test_loader,
        encoder,
        rel_rec,
        rel_send,
        true_edges,
        args.save_dir,
        args,
    )

    print("\n===== Training Complete =====")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()