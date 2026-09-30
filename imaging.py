"""
Conversao de imagens do usuario para o tensor de entrada do modelo.

Entrada do modelo: 28x28x1 INT8, HWC, scale = 1/255, zero_point = -128, ou
seja int8 = pixel_uint8 - 128, com o objeto CLARO sobre fundo ESCURO (como no
MNIST / Fashion-MNIST). Foto comum (objeto escuro em fundo claro) e invertida.
"""
import numpy as np
from PIL import Image


def load_file(path):
    """Imagem qualquer (PNG/JPG/BMP...) -> array uint8 em tons de cinza."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        # fundo transparente vira branco (caso tipico de desenho exportado)
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.asarray(img.convert("L"), dtype=np.uint8)


def needs_invert(gray):
    """Fundo claro? Decide pela borda da imagem, que quase sempre e fundo."""
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return border.mean() > 127


def _bbox(gray, thresh=30):
    ys, xs = np.nonzero(gray > thresh)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _resize_fit(gray, box):
    """Redimensiona mantendo a proporcao para caber em box x box."""
    h, w = gray.shape
    s = box / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return np.asarray(Image.fromarray(gray).resize((nw, nh), Image.LANCZOS), dtype=np.uint8)


def to_model_input(gray, preprocess="mnist", invert=None, size=28):
    """Array uint8 qualquer -> array uint8 size x size pronto para quantizar.

    preprocess:
      "mnist": como o MNIST original -- recorta o digito, encaixa em 20x20 e
               centraliza pelo centro de massa numa tela 28x28.
      "fill":  como o Fashion-MNIST -- recorta o objeto e o encaixa em 28x28,
               centralizado pela caixa.
    invert: None = automatico (pela borda).
    """
    g = gray.astype(np.uint8)
    if invert is None:
        invert = needs_invert(g)
    if invert:
        g = 255 - g

    bb = _bbox(g)
    if bb is None:
        return np.zeros((size, size), np.uint8)
    x0, y0, x1, y1 = bb
    g = g[y0:y1, x0:x1]

    out = np.zeros((size, size), np.uint8)
    if preprocess == "mnist":
        g = _resize_fit(g, 20)
        h, w = g.shape
        ys, xs = np.indices(g.shape)
        m = g.astype(np.float64).sum() or 1.0
        cy, cx = (ys * g).sum() / m, (xs * g).sum() / m
        oy = int(round(size / 2 - cy))
        ox = int(round(size / 2 - cx))
        oy = min(max(oy, 0), size - h)
        ox = min(max(ox, 0), size - w)
    else:
        g = _resize_fit(g, size)
        h, w = g.shape
        oy, ox = (size - h) // 2, (size - w) // 2
    out[oy:oy + h, ox:ox + w] = g
    return out


def quantize(u8, zero_point=-128):
    """uint8 (0..255) -> bytes INT8 HWC com scale 1/255."""
    q = np.clip(u8.astype(np.int16) + zero_point, -128, 127).astype(np.int8)
    return q.reshape(-1).tobytes()


def dequantize(raw, h=28, w=28, zero_point=-128):
    """bytes INT8 (amostra .bin) -> uint8 h x w para exibir."""
    q = np.frombuffer(raw, dtype=np.int8).astype(np.int16)
    return (q - zero_point).clip(0, 255).astype(np.uint8).reshape(h, w)
