#!/usr/bin/env python3
"""
Gera um pacote de modelo em models/<id>/ a partir da saida de
build_mnist_int8.py (repositorio inference_tcc, pasta out/<dataset>/).

O firmware e compilado aqui, numa copia temporaria de
<neorv32>/sw/tcc/mnist_infer com o model_meta.h do modelo -- a
arvore de trabalho do repositorio neorv32 nao e tocada. Requer
riscv-none-elf-gcc e make no PATH (os mesmos do "make exe" do firmware).

O repositorio neorv32 vem de --neorv32 ou da variavel de ambiente NEORV32_HOME.

Uso:
    python make_package.py <inference_tcc>/out/fashion_mnist --neorv32 <repo neorv32>
    python make_package.py <inference_tcc>/out/mnist --name "MNIST (digitos)"
    python make_package.py <out>/mnist --firmware caminho/neorv32_exe.bin   # sem compilar
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import packages

FIRMWARE_SUBDIR = os.path.join("sw", "tcc", "mnist_infer")

# Padroes para os datasets conhecidos; qualquer campo pode ser sobrescrito na linha de comando
KNOWN = {
    "mnist": dict(
        name="MNIST (dígitos manuscritos)",
        description="CNN INT8 treinada no MNIST: dígitos 0-9 escritos à mão.",
        classes=[str(i) for i in range(10)],
        preprocess="mnist",
    ),
    "fashion_mnist": dict(
        name="Fashion-MNIST (roupas)",
        description="CNN INT8 treinada no Fashion-MNIST: 10 tipos de roupa e calçado.",
        classes=["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco",
                 "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"],
        preprocess="fill",
    ),
}


def parse_expected(path):
    """expected.txt -> [(arquivo, label)]"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"(\S+\.bin)\s+idx=\d+\s+label=(\d+)", line)
            if m:
                out.append((m.group(1), int(m.group(2))))
    return out


def build_firmware(neorv32_home, model_meta_h, dest):
    """Compila o firmware com outro model_meta.h numa copia temporaria."""
    tmp = tempfile.mkdtemp(prefix="neorv32_pkg_")
    try:
        src = os.path.join(tmp, "fw")
        shutil.copytree(os.path.join(neorv32_home, FIRMWARE_SUBDIR), src,
                        ignore=shutil.ignore_patterns("build", "*.bin", "*.elf", "*.o"))
        shutil.copy(model_meta_h, os.path.join(src, "model_meta.h"))
        home = os.path.abspath(neorv32_home).replace("\\", "/")
        cmd = ["make", f"NEORV32_HOME={home}", "exe"]
        print("[build]", " ".join(cmd))
        r = subprocess.run(cmd, cwd=src, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stdout.write(r.stdout)
            sys.stderr.write(r.stderr)
            sys.exit("[ERRO] falha ao compilar o firmware")
        shutil.copy(os.path.join(src, "neorv32_exe.bin"), dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="pasta out/<dataset> gerada por build_mnist_int8.py")
    ap.add_argument("--id", help="nome da pasta do pacote (padrao: nome da pasta src)")
    ap.add_argument("--name", help="nome exibido no app")
    ap.add_argument("--description")
    ap.add_argument("--classes", help="nomes das classes separados por virgula")
    ap.add_argument("--preprocess", choices=["mnist", "fill"],
                    help="mnist: digito centrado em 20x20; fill: objeto ocupa 28x28")
    ap.add_argument("--neorv32", default=os.environ.get("NEORV32_HOME"),
                    help="raiz do repositorio neorv32 (padrao: $NEORV32_HOME)")
    ap.add_argument("--firmware", help="neorv32_exe.bin ja compilado (pula a compilacao)")
    ap.add_argument("--models-dir", default=packages.MODELS_DIR)
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dataset = os.path.basename(os.path.normpath(src))
    pkg_id = args.id or dataset
    base = KNOWN.get(dataset, {})
    classes = args.classes.split(",") if args.classes else base.get("classes")
    if not classes:
        sys.exit(f"[ERRO] dataset '{dataset}' desconhecido: informe --classes")

    if not args.firmware:
        if not args.neorv32:
            sys.exit("[ERRO] informe --neorv32 <repo neorv32> (ou defina NEORV32_HOME), ou --firmware")
        if not os.path.isdir(os.path.join(args.neorv32, FIRMWARE_SUBDIR)):
            sys.exit(f"[ERRO] {FIRMWARE_SUBDIR} nao encontrado em {args.neorv32}")

    for f in ("weights.bin", "model_meta.h"):
        if not os.path.isfile(os.path.join(src, f)):
            sys.exit(f"[ERRO] {f} nao encontrado em {src}")

    dst = os.path.join(args.models_dir, pkg_id)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "samples"))

    if args.firmware:
        shutil.copy(args.firmware, os.path.join(dst, "firmware.bin"))
    else:
        build_firmware(args.neorv32, os.path.join(src, "model_meta.h"), os.path.join(dst, "firmware.bin"))
    shutil.copy(os.path.join(src, "weights.bin"), os.path.join(dst, "weights.bin"))
    shutil.copy(os.path.join(src, "model_meta.h"), os.path.join(dst, "model_meta.h"))  # referencia

    samples = []
    expected = os.path.join(src, "expected.txt")
    if os.path.isfile(expected):
        for fname, label in parse_expected(expected):
            p = os.path.join(src, "images", fname)
            if os.path.isfile(p):
                shutil.copy(p, os.path.join(dst, "samples", fname))
                name = classes[label] if label < len(classes) else str(label)
                samples.append({"name": f"{name} ({fname})", "file": f"samples/{fname}", "label": label})

    meta = {
        "name": args.name or base.get("name", dataset),
        "description": args.description or base.get("description", ""),
        "classes": classes,
        "input": {"h": 28, "w": 28, "c": 1, "scale": 1 / 255, "zero_point": -128},
        "preprocess": args.preprocess or base.get("preprocess", "mnist"),
        "clock_hz": 50_000_000,
        "samples": samples,
    }
    with open(os.path.join(dst, "model.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    fw = os.path.getsize(os.path.join(dst, "firmware.bin"))
    print(f"[ok] pacote '{pkg_id}' em {dst}: firmware {fw} B, {len(samples)} amostras")


if __name__ == "__main__":
    main()
