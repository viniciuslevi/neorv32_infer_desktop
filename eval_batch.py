#!/usr/bin/env python3
"""
Coleta em lote para o TCC: envia N imagens do conjunto de teste a placa, registra
classe, logits e ciclos por camada, e compara cada uma com a simulacao bit-exata
do firmware feita aqui no host (mesma aritmetica de inference.c + accel_driver.c,
a partir do weights.bin e do model_meta.h do pacote).

Usa o comando 'H' do firmware (inferencia sem esperar KEY1) e, a cada --sw-every
imagens, 'S' (repete a inferencia so na CPU, para o ganho da rede inteira).

Pressupoe a placa ja com o firmware e os pesos do pacote (pelo app, opcao
"Carregar na placa") ou passe --load para gravar tudo daqui (pede KEY0).

    .venv/Scripts/python eval_batch.py --port COM5 --model models/mnist \\
        --data ../inference_tcc/data/mnist --n 200 --sw-every 50 --out mnist.csv

Saidas: o CSV (uma linha por imagem), <out>.log (texto bruto da UART) e um
resumo no terminal.
"""
import argparse
import csv
import gzip
import os
import re
import struct
import sys
import time

import numpy as np

from board import Board
from packages import load_package

LAYER_NAMES = ["conv1", "pool1", "conv2", "pool2", "dense1", "dense2"]
f32 = np.float32


# ---------------------------------------------------------------------------
# Referencia: simulacao bit-exata do firmware (mesma de build_mnist_int8.py)
# ---------------------------------------------------------------------------
RE_ROW = re.compile(
    r'\{\s*"(\w+)",\s*(\d+),\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*'
    r'(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*'
    r'([-\d.e+]+)f,\s*(-?\d+),\s*([-\d.e+]+)f,\s*([-\d.e+]+)f,\s*(-?\d+),\s*(\d+)\s*\}')


def load_reference_layers(meta_path, weights_path):
    with open(meta_path, encoding="utf-8") as f:
        rows = RE_ROW.findall(f.read())
    blob = open(weights_path, "rb").read()
    layers = []
    for r in rows:
        k, n = int(r[2]), int(r[3])
        w_off, w_bytes, b_off, b_bytes = map(int, r[7:11])
        layers.append(dict(
            name=r[0], k=k, n=n,
            wq=np.frombuffer(blob, np.int8, w_bytes, w_off).reshape(k, n),
            bq=np.frombuffer(blob, "<i4", b_bytes // 4, b_off),
            in_scale=f32(r[11]), in_zp=int(r[12]), w_scale=f32(r[13]),
            out_scale=f32(r[14]), out_zp=int(r[15]), relu=int(r[16])))
    if len(layers) != 4:
        raise SystemExit(f"{meta_path}: esperava 4 camadas, achei {len(layers)}")
    return layers


def _im2col(x_hwc, zp):
    H, W, C = x_hwc.shape
    p = np.pad(x_hwc, ((1, 1), (1, 1), (0, 0)), constant_values=zp)
    cols = [p[dy:dy + H, dx:dx + W, :] for dy in range(3) for dx in range(3)]
    return np.concatenate(cols, axis=2).reshape(H * W, 9 * C)


def _matmul_layer(a_i8, L):
    acc = (a_i8.astype(np.int64) - L["in_zp"]) @ L["wq"].astype(np.int64) + L["bq"]
    mult = f32(f32(L["in_scale"] * L["w_scale"]) / L["out_scale"])
    v = acc.astype(np.float32) * mult
    v = np.where(v >= 0, v + f32(0.5), v - f32(0.5)).astype(np.float32)
    q = np.trunc(v).astype(np.int64) + L["out_zp"]
    return np.clip(q, L["out_zp"] if L["relu"] else -128, 127).astype(np.int8)


def _maxpool(x):
    H, W, C = x.shape
    return x.reshape(H // 2, 2, W // 2, 2, C).max(axis=(1, 3))


def reference_logits(img_i8, layers):
    x = img_i8.reshape(28, 28, 1)
    x = _maxpool(_matmul_layer(_im2col(x, layers[0]["in_zp"]), layers[0]).reshape(28, 28, 32))
    x = _maxpool(_matmul_layer(_im2col(x, layers[1]["in_zp"]), layers[1]).reshape(14, 14, 64))
    x = _matmul_layer(x.reshape(1, -1), layers[2])
    return _matmul_layer(x, layers[3]).reshape(-1)


# ---------------------------------------------------------------------------
# Dados de teste (formato IDX do MNIST/Fashion-MNIST)
# ---------------------------------------------------------------------------
def load_test_set(data_dir):
    with gzip.open(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "rb") as f:
        raw = f.read()
    n, h, w = struct.unpack(">III", raw[4:16])
    x = np.frombuffer(raw, np.uint8, offset=16).reshape(n, h, w)
    with gzip.open(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8).astype(int)
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--model", required=True, help="pasta do pacote (ex.: models/mnist)")
    ap.add_argument("--data", required=True, help="pasta com t10k-*-idx*-ubyte.gz")
    ap.add_argument("--n", type=int, default=100, help="quantas imagens (as N primeiras do teste, ou N sorteadas com --seed)")
    ap.add_argument("--start", type=int, default=0, help="indice da primeira imagem")
    ap.add_argument("--seed", type=int, default=None,
                    help="sorteia N imagens do teste inteiro com esta semente (em ordem crescente "
                         "de indice), em vez de pegar as N primeiras a partir de --start")
    ap.add_argument("--sw-every", type=int, default=0,
                    help="a cada K imagens usa 'S' (tambem so em software); 0 = nunca")
    ap.add_argument("--load", action="store_true", help="grava firmware e pesos antes (pede KEY0)")
    ap.add_argument("--out", required=True, help="CSV de saida")
    args = ap.parse_args()

    pkg = load_package(os.path.abspath(args.model))
    layers = load_reference_layers(os.path.join(pkg.dir, "model_meta.h"), pkg.weights_path)
    x_test, y_test = load_test_set(args.data)
    if args.seed is None:
        idxs = range(args.start, min(args.start + args.n, len(x_test)))
    else:
        rng = np.random.default_rng(args.seed)
        idxs = sorted(int(i) for i in rng.choice(len(x_test), size=args.n, replace=False))

    log_f = open(os.path.splitext(args.out)[0] + ".log", "w", encoding="utf-8")

    def log(s):
        log_f.write(s)
        log_f.flush()
        if s.lstrip().startswith("[app]"):  # avisos do proprio app (ex.: pressione KEY0) vao a tela
            print(s.strip(), flush=True)

    board = Board(args.port, log=log)
    board.open()
    if args.load:
        board.upload_firmware(pkg.read_firmware(), progress=lambda i, n: print(f"\rfirmware {i}/{n}", end=""))
        print()
        board.wait_handshake()
        board.send_weights(pkg.read_weights(), progress=lambda i, n: print(f"\rpesos {i}/{n}", end=""))
        print()
    else:
        # Depois de uma inferencia pelo app (comando 'G') o firmware fica parado
        # esperando KEY1 e nao responde ao handshake ate o botao ser apertado.
        print("[app] Aguardando a placa. Se o LCD mostrar 'KEY1 = proxima', aperte KEY1.", flush=True)

    fields = (["idx", "label", "hw_class", "ref_class", "hw_correct", "ref_correct",
               "class_eq_ref", "logits_eq_ref", "hw_logits", "ref_logits", "total_cycles", "faults",
               "giveups"]
              + [f"cyc_{s}" for s in LAYER_NAMES]
              + ["sw_logits_eq_hw", "total_cycles_sw"] + [f"cyc_sw_{s}" for s in LAYER_NAMES])
    stats = dict(n=0, hw_ok=0, ref_ok=0, cls_eq=0, log_eq=0)
    t_start = time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for j, i in enumerate(idxs):
            img = (x_test[i].reshape(-1).astype(np.int16) - 128).astype(np.int8)
            ref = reference_logits(img, layers)
            ref_cls = int(np.argmax(ref))
            use_sw = args.sw_every > 0 and j % args.sw_every == 0
            board.wait_handshake(timeout_s=60)
            r = board.run(img.tobytes(), timeout_s=600 if use_sw else 120, cmd=b"S" if use_sw else b"H")

            label = int(y_test[i])
            hw_logits = r["logits"] or []
            row = dict(idx=i, label=label, hw_class=r["class"], ref_class=ref_cls,
                       hw_correct=int(r["class"] == label), ref_correct=int(ref_cls == label),
                       class_eq_ref=int(r["class"] == ref_cls),
                       logits_eq_ref=int(hw_logits == [int(v) for v in ref]),
                       hw_logits=" ".join(map(str, hw_logits)), ref_logits=" ".join(str(int(v)) for v in ref),
                       total_cycles=r["total"], faults=r["faults"], giveups=r["giveups"])
            row.update({f"cyc_{s}": r["cycles"].get(s) for s in LAYER_NAMES})
            if use_sw:
                row["sw_logits_eq_hw"] = int(r["logits_sw"] == hw_logits)
                row["total_cycles_sw"] = r["total_sw"]
                row.update({f"cyc_sw_{s}": r["cycles_sw"].get(s) for s in LAYER_NAMES})
            wr.writerow(row)
            f.flush()

            stats["n"] += 1
            stats["hw_ok"] += row["hw_correct"]
            stats["ref_ok"] += row["ref_correct"]
            stats["cls_eq"] += row["class_eq_ref"]
            stats["log_eq"] += row["logits_eq_ref"]
            print(f"[{j + 1}/{len(idxs)}] idx={i} label={label} hw={r['class']} ref={ref_cls}"
                  f" logits={'=' if row['logits_eq_ref'] else 'DIFEREM'}"
                  + (f" sw/hw={r['total_sw'] / r['total']:.2f}x" if use_sw and r["total_sw"] else ""))

    n = stats["n"]
    print(f"\n{n} imagens em {time.time() - t_start:.0f} s")
    print(f"acuracia na placa:          {stats['hw_ok']}/{n} = {100 * stats['hw_ok'] / n:.2f}%")
    print(f"acuracia da referencia:     {stats['ref_ok']}/{n} = {100 * stats['ref_ok'] / n:.2f}%")
    print(f"classe igual a referencia:  {stats['cls_eq']}/{n}")
    print(f"logits identicos a ref.:    {stats['log_eq']}/{n}")
    print(f"desistencias de retry (acumulado desde o boot): {row['giveups']}"
          + ("  <-- dados perdidos, resultados suspeitos" if row["giveups"] else ""))
    log_f.close()
    board.close()


if __name__ == "__main__":
    sys.exit(main())
