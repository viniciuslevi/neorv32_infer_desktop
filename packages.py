"""
Pacotes de modelo: uma pasta por modelo em models/, gerada por make_package.py.

    models/<id>/
        model.json     metadados (nome, classes, entrada, amostras)
        firmware.bin   neorv32_exe.bin compilado com o model_meta.h do modelo
        weights.bin    pesos INT8 + bias INT32 (carregados na SDRAM)
        samples/*.bin  imagens de teste, 784 B INT8 HWC

O firmware e parte do pacote porque o model_meta.h (escalas, zero-points,
camadas) e compilado nele: trocar so os pesos daria resultados errados.
"""
import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")


@dataclass
class Sample:
    name: str
    path: str
    label: int


@dataclass
class ModelPackage:
    id: str
    dir: str
    name: str
    description: str
    classes: list
    input_h: int
    input_w: int
    input_c: int
    input_zero_point: int
    preprocess: str          # "mnist" (digito centrado em 20x20) ou "fill" (objeto ocupa 28x28)
    clock_hz: int
    samples: list = field(default_factory=list)

    @property
    def firmware_path(self):
        return os.path.join(self.dir, "firmware.bin")

    @property
    def weights_path(self):
        return os.path.join(self.dir, "weights.bin")

    @property
    def input_bytes(self):
        return self.input_h * self.input_w * self.input_c

    def read_firmware(self):
        with open(self.firmware_path, "rb") as f:
            return f.read()

    def read_weights(self):
        with open(self.weights_path, "rb") as f:
            return f.read()

    def class_name(self, idx):
        return self.classes[idx] if 0 <= idx < len(self.classes) else f"#{idx}"


def load_package(pkg_dir):
    with open(os.path.join(pkg_dir, "model.json"), encoding="utf-8") as f:
        meta = json.load(f)
    inp = meta["input"]
    return ModelPackage(
        id=os.path.basename(pkg_dir),
        dir=pkg_dir,
        name=meta["name"],
        description=meta.get("description", ""),
        classes=meta["classes"],
        input_h=inp["h"], input_w=inp["w"], input_c=inp["c"],
        input_zero_point=inp["zero_point"],
        preprocess=meta.get("preprocess", "mnist"),
        clock_hz=meta.get("clock_hz", 50_000_000),
        samples=[Sample(s["name"], os.path.join(pkg_dir, s["file"]), s["label"])
                 for s in meta.get("samples", [])],
    )


def discover(models_dir=MODELS_DIR):
    """Todos os pacotes validos em models_dir; pastas quebradas sao ignoradas."""
    pkgs = []
    if not os.path.isdir(models_dir):
        return pkgs
    for entry in sorted(os.listdir(models_dir)):
        d = os.path.join(models_dir, entry)
        if not os.path.isfile(os.path.join(d, "model.json")):
            continue
        try:
            p = load_package(d)
        except (OSError, KeyError, ValueError):
            continue
        if os.path.isfile(p.firmware_path) and os.path.isfile(p.weights_path):
            pkgs.append(p)
    return pkgs
