from __future__ import print_function

import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from boolode_utils import load_boolode_data
from modules import MLPEncoder, MLPDecoder
from utils import (
    my_softmax,
    gumbel_softmax,
    nll_gaussian,
    kl_categorical_uniform,
)


def create_rel_rec_send(num_atoms):
    """
    全ての非自己エッジを表す行列を作成する。

    18遺伝子の場合:
        18 * 17 = 306エッジ
    """

    off_diag = np.ones(
        [num_atoms, num_atoms]
    ) - np.eye(num_atoms)

    receivers = np.where(off_diag)[0]
    senders = np.where(off_diag)[1]

    rel_rec = np.zeros(
        [len(receivers), num_atoms]
    )

    rel_send = np.zeros(
        [len(senders), num_atoms]
    )

    rel_rec[
        np.arange(len(receivers)),
        receivers
    ] = 1

    rel_send[
        np.arange(len(senders)),
        senders
    ] = 1

    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    return rel_rec, rel_send


def calculate_loss(
    data,
    encoder,
    decoder,
    rel_rec,
    rel_send,
    args,
):
    """
    1バッチ分の損失を計算する。
    """

    # Encoderの入力:
    # [batch, genes, timesteps, dimensions]
    logits = encoder(
        data,
        rel_rec,
        rel_send
    )

    # logits:
    # [batch, edges, edge_types]

    edges = gumbel_softmax(
        logits,
        tau=args.temp,
        hard=args.hard
    )

    # 確率分布
    prob = torch.softmax(
        logits,
        dim=-1
    )

    # Decoderによる次時刻予測
    output = decoder(
        data,
        edges,
        rel_rec,
        rel_send,
        pred_steps=args.prediction_steps
    )

    # 正解データは1時刻先
    target = data[:, :, 1:, :]

    # 出力と正解の時間長を合わせる
    min_timesteps = min(
        output.size(2),
        target.size(2)
    )

    output = output[:, :, :min_timesteps, :]
    target = target[:, :, :min_timesteps, :]

    # NLL損失
    loss_nll = nll_gaussian(
        output,
        target,
        args.var
    )

    # KL損失
    loss_kl = kl_categorical_uniform(
        prob,
        args.num_atoms,
        args.edge_types
    )

    loss = loss_nll + loss_kl

    return loss, loss_nll, loss_kl, logits


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

    for batch_idx, batch in enumerate(train_loader):

        # BoolODEのDataLoaderはTensorDataset(data)なので、
        # batchは(data,)というタプルになっている
        data = batch[0]

        if args.cuda:
            data = data.cuda()

        optimizer.zero_grad()

        loss, loss_nll, loss_kl, logits = calculate_loss(
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

        if batch_idx % 10 == 0:
            print(
                "  Batch: {:04d} | Loss: {:.6f} | NLL: {:.6f} | KL: {:.6f}".format(
                    batch_idx,
                    loss.item(),
                    loss_nll.item(),
                    loss_kl.item(),
                )
            )

    return (
        total_loss / num_batches,
        total_nll / num_batches,
        total_kl / num_batches,
    )


def evaluate(
    valid_loader,
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

        for batch in valid_loader:

            data = batch[0]

            if args.cuda:
                data = data.cuda()

            loss, loss_nll, loss_kl, logits = calculate_loss(
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


def save_test_logits(
    test_loader,
    encoder,
    rel_rec,
    rel_send,
    args,
    save_dir,
):
    """
    テストデータに対するEncoderの出力を保存する。
    """

    encoder.eval()

    all_logits = []

    with torch.no_grad():

        for batch in test_loader:

            data = batch[0]

            if args.cuda:
                data = data.cuda()

            logits = encoder(
                data,
                rel_rec,
                rel_send
            )

            all_logits.append(
                logits.cpu().numpy()
            )

    all_logits = np.concatenate(
        all_logits,
        axis=0
    )

    # logitsの形状:
    # [test_samples, edges, edge_types]
    np.save(
        os.path.join(save_dir, "test_logits.npy"),
        all_logits
    )

    # 最も確率の高いエッジタイプ
    predicted_edges = np.argmax(
        all_logits,
        axis=-1
    )

    np.save(
        os.path.join(save_dir, "test_predicted_edges.npy"),
        predicted_edges
    )

    print("")
    print("===== Test logits保存 =====")
    print("logits shape:", all_logits.shape)
    print("predicted edges shape:", predicted_edges.shape)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--encoder-hidden",
        type=int,
        default=256
    )

    parser.add_argument(
        "--decoder-hidden",
        type=int,
        default=256
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--num-atoms",
        type=int,
        default=18
    )

    parser.add_argument(
        "--edge-types",
        type=int,
        default=2
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=899
    )

    parser.add_argument(
        "--dims",
        type=int,
        default=1
    )

    parser.add_argument(
        "--prediction-steps",
        type=int,
        default=1
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005
    )

    parser.add_argument(
        "--var",
        type=float,
        default=5e-5
    )

    parser.add_argument(
        "--hard",
        action="store_true"
    )

    parser.add_argument(
        "--no-factor",
        action="store_true"
    )

    parser.add_argument(
        "--cuda",
        action="store_true"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="logs_boolode/dyn-LL-1"
    )

    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDAが指定されましたが、利用できません。"
        )

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    print("===== BoolODE NRI 学習 =====")
    print("遺伝子数:", args.num_atoms)
    print("時系列長:", args.timesteps)
    print("特徴量数:", args.dims)
    print("エッジタイプ数:", args.edge_types)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("CUDA:", args.cuda)

    # データ読み込み
    train_loader, valid_loader, test_loader = load_boolode_data(
        data_dir="boolode_data/processed/dyn-LL-1",
        batch_size=args.batch_size,
    )

    # エッジ候補行列
    rel_rec, rel_send = create_rel_rec_send(
        args.num_atoms
    )

    if args.cuda:
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()

    print("")
    print("rel_rec shape:", rel_rec.shape)
    print("rel_send shape:", rel_send.shape)

    # Encoder
    encoder = MLPEncoder(
        n_in=args.timesteps * args.dims,
        n_hid=args.encoder_hidden,
        n_out=args.edge_types,
        do_prob=0.0,
        factor=not args.no_factor,
    )

    # Decoder
    decoder = MLPDecoder(
        n_in_node=args.dims,
        edge_types=args.edge_types,
        msg_hid=args.decoder_hidden,
        msg_out=args.decoder_hidden,
        n_hid=args.decoder_hidden,
        do_prob=0.0,
        skip_first=False,
    )

    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    optimizer = optim.Adam(
        list(encoder.parameters())
        + list(decoder.parameters()),
        lr=args.lr
    )

    best_valid_loss = float("inf")
    train_logs = []

    for epoch in range(args.epochs):

        print("")
        print("===== Epoch {} / {} =====".format(
            epoch + 1,
            args.epochs
        ))

        train_loss, train_nll, train_kl = train_one_epoch(
            train_loader,
            encoder,
            decoder,
            optimizer,
            rel_rec,
            rel_send,
            args,
        )

        valid_loss, valid_nll, valid_kl = evaluate(
            valid_loader,
            encoder,
            decoder,
            rel_rec,
            rel_send,
            args,
        )

        print("")
        print(
            "Epoch: {:04d} | "
            "Train Loss: {:.6f} | "
            "Train NLL: {:.6f} | "
            "Train KL: {:.6f}".format(
                epoch + 1,
                train_loss,
                train_nll,
                train_kl,
            )
        )

        print(
            "Epoch: {:04d} | "
            "Valid Loss: {:.6f} | "
            "Valid NLL: {:.6f} | "
            "Valid KL: {:.6f}".format(
                epoch + 1,
                valid_loss,
                valid_nll,
                valid_kl,
            )
        )

        train_logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_nll": train_nll,
            "train_kl": train_kl,
            "valid_loss": valid_loss,
            "valid_nll": valid_nll,
            "valid_kl": valid_kl,
        })

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            torch.save(
                encoder.state_dict(),
                os.path.join(
                    args.save_dir,
                    "encoder_best.pt"
                )
            )

            torch.save(
                decoder.state_dict(),
                os.path.join(
                    args.save_dir,
                    "decoder_best.pt"
                )
            )

            print("Best modelを保存しました。")

    # 設定保存
    with open(
        os.path.join(args.save_dir, "config.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=4,
            ensure_ascii=False
        )

    # 学習ログ保存
    with open(
        os.path.join(args.save_dir, "train_log.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            train_logs,
            f,
            indent=4,
            ensure_ascii=False
        )

    # テストデータに対する関係推定結果を保存
    save_test_logits(
        test_loader,
        encoder,
        rel_rec,
        rel_send,
        args,
        args.save_dir,
    )

    print("")
    print("===== 学習完了 =====")
    print("保存先:", args.save_dir)


if __name__ == "__main__":
    main()
