"""
Comunicacao com a placa DE2-115 (NEORV32 + acelerador sistolico) pela UART.

Duas fases, dois baud rates:
  1. Bootloader do NEORV32 (19200, fixo em sw/bootloader/config.h): grava o
     firmware do modelo (neorv32_exe.bin) -- mesmo protocolo de
     sw/tcc/tools/uart_upload.py.
  2. Firmware sw/tcc/mnist_infer (APP_UART_BAUD = 115200 em main.c): recebe
     pesos e imagem e roda a inferencia -- mesmo protocolo de
     sw/tcc/mnist_infer/tools/send_weights.py.

Diferenca importante em relacao ao send_weights.py: o 'R' de "pronto" so e
aceito DEPOIS de o firmware anunciar que entrou no handshake. Antes disso ele
imprime o diagnostico da SDRAM e o benchmark, textos cheios de 'R' maiusculo
("NEORV32", "UART", ...) que seriam confundidos com o sinal de pronto.

Este modulo nao depende de Qt: os callbacks `log` e `progress` sao chamados na
thread de quem invoca, e `cancel` e consultado nos lacos longos.
"""
import re
import struct
import time

import serial
from serial.tools import list_ports

BOOT_BAUD = 19200
APP_BAUD = 115200

# Bootloader: envio pausado, 16 bytes a cada 10 ms (ver uart_upload.py -- o
# bootloader nao absorve rajada continua, nem com adaptador FTDI).
BOOT_CHUNK = 16
BOOT_CHUNK_DELAY_S = 0.01

# Firmware: cada pedaco de 64 bytes e confirmado com 'K' ou rejeitado com 'N'
# (sdram.c: sdram_uart_recv_blob, BLOB_CHUNK_SIZE).
BLOB_CHUNK = 64
BLOB_ACK_TIMEOUT_S = 3
BLOB_MAX_RETRIES = 20

# Linhas que o firmware imprime imediatamente antes de entrar/voltar ao laco
# de handshake; so depois delas um 'R' significa "pronto".
MARK_HANDSHAKE = b"Aguardando handshake"
MARK_BLOB_DONE = b"[sdram] "

RE_CLASS = re.compile(rb">>> Classe prevista: (-?\d+) <<<")
RE_LOGITS = re.compile(rb"\[logits\] ((?:-?\d+ ?)+)")
RE_BENCH = re.compile(rb"\[bench\] (\w+)\s*: (\d+) ciclos")
# Comando 'S' (coleta do TCC): a mesma inferencia repetida so na CPU
RE_BENCH_SW = re.compile(rb"\[bench-sw\] (\w+)\s*: (\d+) ciclos")
RE_LOGITS_SW = re.compile(rb"\[logits-sw\] ((?:-?\d+ ?)+)")
RE_TOTAL = re.compile(rb"\[total\] (\d+) ciclos")
RE_TOTAL_SW = re.compile(rb"\[total-sw\] (\d+) ciclos")
RE_FAULTS = re.compile(rb"\[faults ate aqui: (\d+)\]")
RE_GIVEUPS = re.compile(rb"\[faults desistidos: (\d+)\]")
MARK_BATCH_END = b"[fim]"


class BoardError(Exception):
    pass


class Cancelled(BoardError):
    pass


def list_serial_ports():
    """[(device, descricao)] das portas seriais presentes."""
    return [(p.device, p.description) for p in sorted(list_ports.comports(), key=lambda p: p.device)]


def _noop(*_a, **_k):
    pass


class Board:
    def __init__(self, port, log=None, cancel=None):
        self.port = port
        self.log = log or _noop           # log(str): texto recebido da placa / mensagens do app
        self.cancel = cancel or (lambda: False)
        self.ser = None

    # ------------------------------------------------------------------ porta
    def open(self, baud=APP_BAUD):
        if self.ser is None:
            try:
                self.ser = serial.Serial(self.port, baud, timeout=0.1, write_timeout=10)
            except serial.SerialException as e:
                raise BoardError(f"Nao foi possivel abrir {self.port}: {e}") from e
        else:
            self.ser.baudrate = baud

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def _check_cancel(self):
        if self.cancel():
            raise Cancelled("Operacao cancelada")

    def _read_until(self, pattern, timeout_s, echo=True):
        """Le ate `pattern` (bytes ou regex compilada) aparecer; devolve tudo o que leu."""
        buf = b""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._check_cancel()
            data = self.ser.read(self.ser.in_waiting or 1)
            if data:
                buf += data
                if echo:
                    self.log(data.decode(errors="replace"))
                found = pattern.search(buf) if hasattr(pattern, "search") else (pattern in buf)
                if found:
                    return buf
        raise BoardError(f"Tempo esgotado ({timeout_s:.0f}s) esperando {pattern!r} da placa")

    # ------------------------------------------------------- 1. bootloader
    def upload_firmware(self, exe: bytes, progress=_noop, wait_reset_s=60):
        """Espera o usuario resetar a placa (KEY0), grava `exe` pelo bootloader e o executa."""
        self.open(BOOT_BAUD)
        self.ser.reset_input_buffer()
        self.log("\n[app] Pressione KEY0 (reset) na placa...\n")
        self._read_until(b"Press any key to abort", wait_reset_s)

        self.ser.write(b" u")  # SPACE aborta o auto-boot, 'u' inicia o upload
        self._read_until(b"Awaiting", 5)
        for i in range(0, len(exe), BOOT_CHUNK):
            self._check_cancel()
            self.ser.write(exe[i:i + BOOT_CHUNK])
            self.ser.flush()
            time.sleep(BOOT_CHUNK_DELAY_S)
            progress(min(i + BOOT_CHUNK, len(exe)), len(exe))

        # 10 bits por byte na linha, mais folga
        buf = self._read_until(re.compile(rb"OK|ERR"), len(exe) // (BOOT_BAUD // 10) + 10)
        if b"OK" not in buf:
            raise BoardError("O bootloader rejeitou o executavel (veja o log)")
        self.ser.write(b"e")
        self.ser.flush()
        # O firmware ja comeca a imprimir em APP_BAUD: troca o baud imediatamente
        time.sleep(0.05)
        self.open(APP_BAUD)
        self.ser.reset_input_buffer()

    # ------------------------------------------------------- 2. firmware
    def wait_handshake(self, timeout_s=180, marker=MARK_HANDSHAKE):
        """Espera o firmware estar no laco de handshake (pronto para um comando).

        Reconhece duas situacoes:
          - o firmware acabou de chegar ao laco: aparece a linha `marker` e,
            depois do fim dela, um 'R' (a propria linha pode conter 'R');
          - o firmware ja estava no laco ha tempo (o texto ja passou): chegam so
            'R's isolados, um a cada ~300 ms, sem nenhum outro byte entre eles.
        """
        self.open(APP_BAUD)
        buf = b""
        armed = False          # marcador visto e linha encerrada
        last_lone_r = None     # instante do ultimo 'R' que chegou sozinho
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._check_cancel()
            data = self.ser.read(self.ser.in_waiting or 1)
            if not data:
                continue
            now = time.time()
            if data == b"R":
                if armed or (last_lone_r is not None and now - last_lone_r > 0.2):
                    break
                last_lone_r = now
                continue
            last_lone_r = None
            if armed and b"R" in data:
                break
            buf += data
            self.log(data.decode(errors="replace"))
            i = buf.find(marker)
            if i >= 0:
                nl = buf.find(b"\n", i)
                if nl >= 0:
                    armed = True
                    if b"R" in buf[nl + 1:]:
                        break
        else:
            raise BoardError(f"Tempo esgotado ({timeout_s:.0f}s) esperando a placa ficar pronta")
        # descarta 'R's repetidos (reenviados a cada 300 ms) antes do comando
        self.ser.reset_input_buffer()

    def _send_chunked(self, data, progress):
        total = len(data)
        sent = 0
        while sent < total:
            chunk = data[sent:sent + BLOB_CHUNK]
            for attempt in range(BLOB_MAX_RETRIES):
                self._check_cancel()
                self.ser.write(chunk)
                resp = self._read_ack()
                if resp == b"K":
                    break
                if attempt == BLOB_MAX_RETRIES - 1:
                    raise BoardError(f"Pedaco no offset {sent} falhou apos {BLOB_MAX_RETRIES} tentativas")
            sent += len(chunk)
            progress(sent, total)

    def _read_ack(self):
        """Proximo 'K' ou 'N' (ignora um 'R' atrasado do handshake); b'' se tempo esgotar."""
        deadline = time.time() + BLOB_ACK_TIMEOUT_S
        while time.time() < deadline:
            b = self.ser.read(1)
            if b in (b"K", b"N"):
                return b
        return b""

    def send_blob(self, cmd: bytes, data: bytes, progress=_noop):
        """Envia um comando 'W' (pesos) ou 'I' (imagem). Exige wait_handshake antes."""
        self.ser.write(cmd)
        self.ser.write(struct.pack("<I", len(data)))
        self._send_chunked(data, progress)
        old = self.ser.timeout
        self.ser.timeout = 5
        ack = self.ser.read(4)
        self.ser.timeout = old
        if len(ack) != 4 or struct.unpack("<I", ack)[0] != len(data):
            raise BoardError(f"Confirmacao final de '{cmd.decode()}' invalida: {ack!r}")

    def send_weights(self, weights: bytes, progress=_noop):
        """Grava os pesos na SDRAM. Pressupoe a placa no handshake; termina nele."""
        self.log("\n[app] Enviando pesos...\n")
        self.send_blob(b"W", weights, progress)
        self.wait_handshake(30, marker=MARK_BLOB_DONE)

    def run(self, image: bytes, timeout_s=120, cmd=b"G"):
        """Envia a imagem e o comando; devolve dict com classe, logits e ciclos por camada.

        Pressupoe a placa no handshake. Com 'G' o firmware fica esperando KEY1
        antes de voltar ao handshake. 'H' e 'S' (coleta em lote) voltam direto,
        depois de imprimir "[fim]"; 'S' tambem repete a inferencia so na CPU e
        preenche as chaves "*_sw"."""
        self.log("\n[app] Enviando imagem...\n")
        self.send_blob(b"I", image)
        self.wait_handshake(30, marker=MARK_BLOB_DONE)
        self.ser.write(cmd)
        if cmd == b"G":
            buf = self._read_until(RE_CLASS, timeout_s)
            m = RE_CLASS.search(buf)
            if b"\n" not in buf[m.end():]:
                buf += self._read_until(b"\n", 2)  # termina a linha (o log fica completo)
        else:
            buf = self._read_until(MARK_BATCH_END, timeout_s)

        def ints(regex):
            m = regex.search(buf)
            return [int(v) for v in m.group(1).split()] if m else None

        def one(regex):
            m = regex.search(buf)
            return int(m.group(1)) if m else None

        return {
            "class": int(RE_CLASS.search(buf).group(1)),
            "logits": ints(RE_LOGITS),
            "cycles": {m.group(1).decode(): int(m.group(2)) for m in RE_BENCH.finditer(buf)},
            "total": one(RE_TOTAL),
            "faults": one(RE_FAULTS),
            # ultima ocorrencia: no modo 'S' inclui as desistencias da execucao so em software
            "giveups": int(RE_GIVEUPS.findall(buf)[-1]) if RE_GIVEUPS.search(buf) else None,
            "logits_sw": ints(RE_LOGITS_SW),
            "cycles_sw": {m.group(1).decode(): int(m.group(2)) for m in RE_BENCH_SW.finditer(buf)},
            "total_sw": one(RE_TOTAL_SW),
            "log": buf.decode(errors="replace"),
        }
