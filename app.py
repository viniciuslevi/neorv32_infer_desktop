#!/usr/bin/env python3
"""
NEORV32 Inferência -- app desktop para rodar modelos CNN INT8 na DE2-115.

Fluxo:
  1. escolher a porta serial e o modelo (pacote em models/, ver make_package.py);
  2. "Carregar na placa": grava o firmware do modelo pelo bootloader (pede
     reset) e envia os pesos para a SDRAM;
  3. escolher uma imagem (arquivo, desenho ou amostra do dataset) e "Inferir".

A comunicacao serial roda numa thread separada (Worker) para a janela nao
travar durante os ~422 KB de pesos.
"""
import os
import sys
import threading

import numpy as np
from PySide6.QtCore import QObject, QSettings, QThread, Qt, Signal, Slot, QPoint
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import board as brd
import imaging
import packages

PREVIEW_PX = 224  # 28 x 8


# --------------------------------------------------------------------------- worker
class Worker(QObject):
    """Executa as operacoes bloqueantes da serial fora da thread da interface."""
    log = Signal(str)
    progress = Signal(int, int)
    status = Signal(str)
    done = Signal(str, object)     # (operacao, resultado)
    failed = Signal(str, str)      # (operacao, mensagem)

    def __init__(self):
        super().__init__()
        self.cancel_flag = threading.Event()
        self.board = None

    def _board(self, port):
        if self.board is not None and self.board.port != port:
            self.board.close()
            self.board = None
        if self.board is None:
            self.board = brd.Board(port, log=self.log.emit, cancel=self.cancel_flag.is_set)
        return self.board

    @Slot(str, object)
    def load_model(self, port, pkg):
        try:
            b = self._board(port)
            self.status.emit("Pressione KEY0 (reset) na placa...")
            fw = pkg.read_firmware()
            b.upload_firmware(fw, progress=lambda i, n: self.progress.emit(i, n))
            self.status.emit("Firmware gravado. Aguardando diagnóstico da SDRAM...")
            self.progress.emit(0, 0)
            b.wait_handshake()
            self.status.emit("Enviando pesos (115200 baud)...")
            b.send_weights(pkg.read_weights(), progress=lambda i, n: self.progress.emit(i, n))
            self.done.emit("load", pkg)
        except Exception as e:  # noqa: BLE001 -- qualquer falha vira mensagem na interface
            self._fail("load", e)

    @Slot(str, bool, object)
    def infer(self, port, ready, image):
        try:
            b = self._board(port)
            if not ready:
                self.status.emit("Aguardando a placa (pressione KEY1 se ela mostrar um resultado)...")
                self.progress.emit(0, 0)
                b.wait_handshake(timeout_s=600)
            self.status.emit("Inferindo...")
            self.progress.emit(0, 0)
            self.done.emit("infer", b.run(image))
        except Exception as e:  # noqa: BLE001
            self._fail("infer", e)

    @Slot()
    def close(self):
        if self.board is not None:
            self.board.close()
            self.board = None

    def _fail(self, op, e):
        # depois de erro/cancelamento a serial pode ter bytes pela metade: reabre limpa
        self.close()
        self.failed.emit(op, str(e) if isinstance(e, brd.BoardError) else f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- widgets
def gray_to_pixmap(gray, size=PREVIEW_PX):
    g = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = g.shape
    img = QImage(g.data, w, h, w, QImage.Format_Grayscale8).copy()
    return QPixmap.fromImage(img.scaled(size, size, Qt.KeepAspectRatio, Qt.FastTransformation))


class DrawCanvas(QWidget):
    """Tela de desenho: traco branco sobre fundo preto, como no MNIST."""
    changed = Signal()

    def __init__(self, size=280, pen=20):
        super().__init__()
        self.setFixedSize(size, size)
        self.pen_w = pen
        self.image = QImage(size, size, QImage.Format_Grayscale8)
        self.image.fill(0)
        self.last = None
        self.setCursor(Qt.CrossCursor)

    def clear(self):
        self.image.fill(0)
        self.update()
        self.changed.emit()

    def to_gray(self):
        img = self.image
        buf = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes())
        return buf.reshape(img.height(), img.bytesPerLine())[:, :img.width()].copy()

    def is_empty(self):
        return not self.to_gray().any()

    def paintEvent(self, _):
        p = QPainter(self)
        p.drawImage(0, 0, self.image)

    def _line(self, a, b):
        p = QPainter(self.image)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255), self.pen_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(a, b)
        p.end()
        self.update()

    def mousePressEvent(self, e):
        self.last = e.position().toPoint()
        self._line(self.last, self.last + QPoint(0, 0))

    def mouseMoveEvent(self, e):
        if self.last is not None:
            pt = e.position().toPoint()
            self._line(self.last, pt)
            self.last = pt

    def mouseReleaseEvent(self, _):
        self.last = None
        self.changed.emit()


class ScoreBars(QWidget):
    """Logits INT8 por classe, com a classe prevista destacada."""

    def __init__(self):
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setVerticalSpacing(2)
        self.rows = []

    def show_logits(self, classes, logits, pred):
        while self.grid.count():
            self.grid.takeAt(0).widget().deleteLater()
        for i, (name, v) in enumerate(zip(classes, logits)):
            lab = QLabel(name)
            bar = QProgressBar()
            bar.setRange(-128, 127)
            bar.setValue(v)
            bar.setFormat(str(v))
            bar.setMaximumHeight(16)
            if i == pred:
                f = lab.font(); f.setBold(True); lab.setFont(f)
                bar.setStyleSheet("QProgressBar::chunk { background: #2e7d32; }")
            self.grid.addWidget(lab, i, 0)
            self.grid.addWidget(bar, i, 1)


# --------------------------------------------------------------------------- janela
class MainWindow(QMainWindow):
    sig_load = Signal(str, object)
    sig_infer = Signal(str, bool, object)
    sig_close = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEORV32 Inferência — DE2-115")
        self.settings = QSettings("univasf", "neorv32_infer_desktop")

        self.packages = packages.discover()
        self.loaded_pkg = None     # modelo presente na placa (nesta sessao)
        self.board_ready = False   # placa no handshake, sem esperar KEY1
        self.busy = False
        self.input_bytes = None    # tensor INT8 que sera enviado
        self.input_label = None    # rotulo verdadeiro, quando conhecido (amostra)
        self.file_gray = None      # imagem aberta do disco (antes do pre-processamento)

        self._build_ui()
        self._start_worker()
        self.refresh_ports()
        self.on_model_changed()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        # conexao
        top = QHBoxLayout()
        top.addWidget(QLabel("Porta serial:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(260)
        top.addWidget(self.port_combo)
        b = QPushButton("Atualizar")
        b.clicked.connect(self.refresh_ports)
        top.addWidget(b)
        top.addStretch()
        self.board_label = QLabel()
        top.addWidget(self.board_label)
        root.addLayout(top)

        cols = QHBoxLayout()
        cols.addWidget(self._build_model_box(), 1)
        cols.addWidget(self._build_image_box(), 2)
        cols.addWidget(self._build_result_box(), 2)
        body = QWidget()
        body.setLayout(cols)

        # log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setFont(QFont("Consolas", 9))
        log_box = QGroupBox("Saída da placa (UART)")
        lv = QVBoxLayout(log_box)
        lv.addWidget(self.log_view)

        split = QSplitter(Qt.Vertical)
        split.addWidget(body)
        split.addWidget(log_box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        # barra de status
        bottom = QHBoxLayout()
        self.status_label = QLabel("Pronto.")
        bottom.addWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(300)
        self.progress.setVisible(False)
        bottom.addWidget(self.progress)
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.on_cancel)
        bottom.addWidget(self.cancel_btn)
        root.addLayout(bottom)

        self.setCentralWidget(central)
        self.setAcceptDrops(True)
        self.resize(1200, 780)
        self._update_board_label()

    def _build_model_box(self):
        box = QGroupBox("1. Modelo")
        v = QVBoxLayout(box)
        self.model_combo = QComboBox()
        for p in self.packages:
            self.model_combo.addItem(p.name, p)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        v.addWidget(self.model_combo)
        self.model_info = QLabel()
        self.model_info.setWordWrap(True)
        self.model_info.setAlignment(Qt.AlignTop)
        v.addWidget(self.model_info)
        v.addStretch()
        self.load_btn = QPushButton("Carregar na placa")
        self.load_btn.setToolTip("Grava o firmware do modelo (pede reset) e envia os pesos para a SDRAM")
        self.load_btn.clicked.connect(self.on_load)
        v.addWidget(self.load_btn)
        self.assume_btn = QPushButton("A placa já está com este modelo")
        self.assume_btn.setFlat(True)
        self.assume_btn.setToolTip("Pula a gravação: use quando o firmware e os pesos deste modelo já estão na placa")
        self.assume_btn.clicked.connect(self.on_assume_loaded)
        v.addWidget(self.assume_btn)
        if not self.packages:
            self.model_info.setText(
                "Nenhum modelo encontrado em models/.\n\nGere os pacotes com:\n"
                "python make_package.py <inference_tcc>/out/mnist")
            self.load_btn.setEnabled(False)
            self.assume_btn.setEnabled(False)
        return box

    def _build_image_box(self):
        box = QGroupBox("2. Imagem")
        v = QVBoxLayout(box)
        self.tabs = QTabWidget()

        # arquivo
        t_file = QWidget()
        fv = QVBoxLayout(t_file)
        b = QPushButton("Abrir imagem...")
        b.clicked.connect(self.on_open_file)
        fv.addWidget(b)
        self.file_label = QLabel("PNG, JPG ou BMP. Também dá para arrastar o arquivo para a janela.")
        self.file_label.setWordWrap(True)
        fv.addWidget(self.file_label)
        fv.addWidget(QLabel("Cores:"))
        self.invert_combo = QComboBox()
        self.invert_combo.addItems(["Automático (fundo claro é invertido)", "Inverter", "Não inverter"])
        self.invert_combo.currentIndexChanged.connect(self.refresh_file_input)
        fv.addWidget(self.invert_combo)
        fv.addStretch()
        self.tabs.addTab(t_file, "Arquivo")

        # desenho
        t_draw = QWidget()
        dv = QVBoxLayout(t_draw)
        self.canvas = DrawCanvas()
        self.canvas.changed.connect(self.refresh_draw_input)
        dv.addWidget(self.canvas, 0, Qt.AlignHCenter)
        b = QPushButton("Limpar")
        b.clicked.connect(self.canvas.clear)
        dv.addWidget(b)
        self.tabs.addTab(t_draw, "Desenhar")

        # amostras
        t_samp = QWidget()
        sv = QVBoxLayout(t_samp)
        self.sample_list = QListWidget()
        self.sample_list.currentItemChanged.connect(self.on_sample_selected)
        sv.addWidget(self.sample_list)
        self.tabs.addTab(t_samp, "Amostras do dataset")

        self.tabs.currentChanged.connect(self.on_tab_changed)
        v.addWidget(self.tabs, 1)

        # previa do que vai para a placa
        v.addWidget(QLabel("Entrada enviada à placa (28×28, INT8):"))
        self.preview = QLabel()
        self.preview.setFixedSize(PREVIEW_PX, PREVIEW_PX)
        self.preview.setFrameShape(QFrame.Box)
        self.preview.setAlignment(Qt.AlignCenter)
        v.addWidget(self.preview, 0, Qt.AlignHCenter)
        return box

    def _build_result_box(self):
        box = QGroupBox("3. Resultado")
        v = QVBoxLayout(box)
        self.infer_btn = QPushButton("Inferir")
        f = self.infer_btn.font(); f.setPointSize(f.pointSize() + 3); f.setBold(True)
        self.infer_btn.setFont(f)
        self.infer_btn.setMinimumHeight(44)
        self.infer_btn.clicked.connect(self.on_infer)
        v.addWidget(self.infer_btn)
        self.result_label = QLabel("—")
        f = self.result_label.font(); f.setPointSize(26); f.setBold(True)
        self.result_label.setFont(f)
        self.result_label.setAlignment(Qt.AlignCenter)
        v.addWidget(self.result_label)
        self.result_detail = QLabel()
        self.result_detail.setAlignment(Qt.AlignCenter)
        self.result_detail.setWordWrap(True)
        v.addWidget(self.result_detail)
        v.addWidget(QLabel("Pontuação por classe (logits INT8):"))
        self.scores = ScoreBars()
        v.addWidget(self.scores)
        v.addStretch()
        return box

    # ---------------------------------------------------------------- worker
    def _start_worker(self):
        self.thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.sig_load.connect(self.worker.load_model)
        self.sig_infer.connect(self.worker.infer)
        self.sig_close.connect(self.worker.close)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.on_progress)
        self.worker.status.connect(self.status_label.setText)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.thread.start()

    def closeEvent(self, e):
        self.worker.cancel_flag.set()
        self.sig_close.emit()
        self.thread.quit()
        self.thread.wait(3000)
        super().closeEvent(e)

    # ---------------------------------------------------------------- estado
    def current_pkg(self):
        return self.model_combo.currentData()

    def port(self):
        return self.port_combo.currentData()

    def set_busy(self, busy):
        self.busy = busy
        for w in (self.load_btn, self.assume_btn, self.infer_btn, self.port_combo, self.model_combo):
            w.setEnabled(not busy and bool(self.packages))
        self.progress.setVisible(busy)
        self.cancel_btn.setVisible(busy)
        if busy:
            self.worker.cancel_flag.clear()
            self.progress.setRange(0, 0)

    def _update_board_label(self):
        if self.loaded_pkg is None:
            txt = "Placa: nenhum modelo carregado nesta sessão"
        else:
            txt = f"Placa: <b>{self.loaded_pkg.name}</b>"
            if not self.board_ready:
                txt += " — aguardando KEY1"
        self.board_label.setText(txt)

    def refresh_ports(self):
        last = self.settings.value("port", "")
        self.port_combo.clear()
        for dev, desc in brd.list_serial_ports():
            self.port_combo.addItem(f"{dev} — {desc}", dev)
            if dev == last:
                self.port_combo.setCurrentIndex(self.port_combo.count() - 1)
        if self.port_combo.count() == 0:
            self.port_combo.addItem("(nenhuma porta encontrada)", None)

    def append_log(self, text):
        self.log_view.moveCursor(QTextCursor.End)
        self.log_view.insertPlainText(text.replace("\r", ""))
        self.log_view.moveCursor(QTextCursor.End)

    # ---------------------------------------------------------------- modelo
    def on_model_changed(self):
        pkg = self.current_pkg()
        if pkg is None:
            return
        self.model_info.setText(
            f"{pkg.description}\n\nClasses: {', '.join(pkg.classes)}\n"
            f"Entrada: {pkg.input_h}×{pkg.input_w}×{pkg.input_c} INT8\n"
            f"Pesos: {os.path.getsize(pkg.weights_path) / 1024:.0f} KB · "
            f"firmware: {os.path.getsize(pkg.firmware_path) / 1024:.0f} KB")
        self.sample_list.clear()
        for s in pkg.samples:
            it = QListWidgetItem(s.name)
            it.setData(Qt.UserRole, s)
            self.sample_list.addItem(it)
        self.refresh_input()

    def _need_port(self):
        if self.port() is None:
            QMessageBox.warning(self, "Porta serial", "Selecione a porta serial da placa.")
            return False
        self.settings.setValue("port", self.port())
        return True

    def on_load(self):
        if not self._need_port():
            return
        pkg = self.current_pkg()
        QMessageBox.information(
            self, "Carregar modelo",
            f"Vou gravar \"{pkg.name}\" na placa.\n\n"
            "Depois de clicar em OK, pressione KEY0 (reset) na DE2-115. "
            "O app detecta o bootloader sozinho.")
        self.loaded_pkg = None
        self.board_ready = False
        self._update_board_label()
        self.set_busy(True)
        self.sig_load.emit(self.port(), pkg)

    def on_assume_loaded(self):
        self.loaded_pkg = self.current_pkg()
        self.board_ready = False  # descobre na hora de inferir se ela ja esta no handshake
        self._update_board_label()
        self.status_label.setText("Ok: o app vai assumir que a placa já tem este modelo.")

    # ---------------------------------------------------------------- imagem
    def _set_input(self, raw, label=None):
        self.input_bytes = raw
        self.input_label = label
        if raw is None:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("sem imagem")
        else:
            pkg = self.current_pkg()
            zp = pkg.input_zero_point if pkg else -128
            self.preview.setPixmap(gray_to_pixmap(imaging.dequantize(raw, zero_point=zp)))

    def _from_gray(self, gray, invert):
        pkg = self.current_pkg()
        pre = pkg.preprocess if pkg else "mnist"
        zp = pkg.input_zero_point if pkg else -128
        return imaging.quantize(imaging.to_model_input(gray, pre, invert=invert), zp)

    def refresh_input(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.refresh_file_input()
        elif idx == 1:
            self.refresh_draw_input()
        else:
            self.on_sample_selected(self.sample_list.currentItem())

    def on_tab_changed(self, _):
        self.refresh_input()

    def on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagem", self.settings.value("img_dir", ""),
            "Imagens (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;Amostra INT8 (*.bin)")
        if path:
            self.open_path(path)

    def open_path(self, path):
        self.settings.setValue("img_dir", os.path.dirname(path))
        self.tabs.setCurrentIndex(0)
        try:
            if path.lower().endswith(".bin"):
                with open(path, "rb") as f:
                    raw = f.read()
                if len(raw) != 784:
                    raise ValueError(f"esperado 784 bytes, o arquivo tem {len(raw)}")
                self.file_gray = imaging.dequantize(raw)
            else:
                self.file_gray = imaging.load_file(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Imagem", f"Não foi possível abrir a imagem:\n{e}")
            return
        self.file_label.setText(os.path.basename(path))
        self.refresh_file_input()

    def refresh_file_input(self):
        if self.tabs.currentIndex() != 0:
            return
        if self.file_gray is None:
            self._set_input(None)
            return
        invert = {0: None, 1: True, 2: False}[self.invert_combo.currentIndex()]
        self._set_input(self._from_gray(self.file_gray, invert))

    def refresh_draw_input(self):
        if self.tabs.currentIndex() != 1:
            return
        if self.canvas.is_empty():
            self._set_input(None)
        else:
            self._set_input(self._from_gray(self.canvas.to_gray(), invert=False))

    def on_sample_selected(self, item, _prev=None):
        if self.tabs.currentIndex() != 2:
            return
        if item is None:
            self._set_input(None)
            return
        s = item.data(Qt.UserRole)
        with open(s.path, "rb") as f:
            self._set_input(f.read(), s.label)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.open_path(urls[0].toLocalFile())

    # ---------------------------------------------------------------- inferencia
    def on_infer(self):
        if self.input_bytes is None:
            QMessageBox.information(self, "Inferir", "Escolha uma imagem primeiro.")
            return
        if self.loaded_pkg is None:
            QMessageBox.information(self, "Inferir",
                                    "Carregue um modelo na placa primeiro (ou use \"A placa já está com este modelo\").")
            return
        if not self._need_port():
            return
        if self.current_pkg() is not self.loaded_pkg:
            r = QMessageBox.question(
                self, "Modelo diferente",
                f"A placa está com \"{self.loaded_pkg.name}\", não com o modelo selecionado.\n"
                "Inferir mesmo assim com o modelo da placa?")
            if r != QMessageBox.Yes:
                return
        self.result_label.setText("…")
        self.result_detail.setText("")
        self.set_busy(True)
        self.sig_infer.emit(self.port(), self.board_ready, self.input_bytes)

    # ---------------------------------------------------------------- retorno do worker
    def on_progress(self, i, n):
        if n == 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, n)
            self.progress.setValue(i)

    def on_cancel(self):
        self.worker.cancel_flag.set()
        self.status_label.setText("Cancelando...")

    def on_done(self, op, result):
        self.set_busy(False)
        if op == "load":
            self.loaded_pkg = result
            self.board_ready = True
            self.status_label.setText(f"Modelo \"{result.name}\" carregado. Escolha uma imagem e clique em Inferir.")
        elif op == "infer":
            self.board_ready = False  # firmware segura o resultado no LCD ate KEY1
            self.show_result(result)
            self.status_label.setText("Pronto. Para a próxima imagem, pressione KEY1 na placa (antes ou depois de clicar em Inferir).")
        self._update_board_label()

    def on_failed(self, op, msg):
        self.set_busy(False)
        if op == "load":
            self.loaded_pkg = None
        self.board_ready = False
        self._update_board_label()
        self.status_label.setText(f"Erro: {msg}")
        if "cancelada" not in msg:
            QMessageBox.warning(self, "Erro", msg)

    def show_result(self, r):
        pkg = self.loaded_pkg
        cls = r["class"]
        self.result_label.setText(pkg.class_name(cls))
        parts = [f"classe {cls}"]
        if r["cycles"]:
            total = sum(r["cycles"].values())
            parts.append(f"{total:,} ciclos = {1000 * total / pkg.clock_hz:.1f} ms a "
                         f"{pkg.clock_hz / 1e6:.0f} MHz".replace(",", "."))
        if self.input_label is not None:
            ok = self.input_label == cls
            parts.append(("✔ acertou" if ok else f"✘ errou (esperado: {pkg.class_name(self.input_label)})"))
        self.result_detail.setText("\n".join(parts))
        tip = "\n".join(f"{k}: {v:,} ciclos".replace(",", ".") for k, v in r["cycles"].items())
        self.result_detail.setToolTip(tip)
        if r["logits"]:
            self.scores.show_logits(pkg.classes, r["logits"], cls)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NEORV32 Inferência")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
