# NEORV32 Inferência (desktop)

Aplicativo desktop e ferramenta de coleta do TCC **"Aceleração de Redes Neurais em
FPGA Cyclone IV E"** (Vinícius Levi dos Santos, Engenharia da Computação, UNIVASF,
2026). Roda modelos CNN INT8 no processador NEORV32 com o acelerador sistólico, na
placa DE2-115: escolhe o modelo, grava na placa, escolhe uma imagem e mostra a
classe prevista, o tempo e a pontuação por classe.

Faz parte de três repositórios:

| Repositório | Papel |
|---|---|
| [`neorv32`](https://github.com/viniciuslevi/neorv32/tree/systolic-array-accelerator) (ramo `systolic-array-accelerator`) | Hardware (acelerador, controlador SDRAM, SoC) e *firmware* `sw/tcc/mnist_infer` |
| [`inference_tcc`](https://github.com/viniciuslevi/inference_tcc) | Treinamento, quantização INT8 e referência bit-exata dos modelos |
| `neorv32_infer_desktop` (este) | Aplicativo de operação da placa e coleta em lote |

Conversa com o firmware `sw/tcc/mnist_infer` pela UART da placa, com o mesmo
protocolo do `send_weights.py` do firmware.

## Instalação

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Testado com Python 3.12.7, PySide6 6.11.2, pySerial 3.5 e NumPy 2.5.3, no Windows 11,
com o adaptador USB-serial Comm5 1S-USB (FTDI) no conector RS-232 da placa.

## Aplicativo

Abra com `abrir.bat` (duplo clique) ou `.venv/Scripts/python app.py`.

1. **Porta serial**: a do adaptador USB-serial da placa.
2. **Modelo** → *Carregar na placa*: depois do OK, pressione **KEY0** (reset).
   O app detecta o bootloader, grava o firmware do modelo (19200 baud), espera
   o diagnóstico da SDRAM e envia os pesos (115200 baud).
   Se a placa já tem o modelo, use *A placa já está com este modelo*.
3. **Imagem**, em uma de três abas:
   - *Arquivo*: PNG/JPG/BMP (ou arraste para a janela). Fundo claro é
     invertido automaticamente; o objeto é recortado e centralizado como no
     dataset.
   - *Desenhar*: desenhe com o mouse.
   - *Amostras do dataset*: imagens de teste com rótulo conhecido (mostra se acertou).

   O quadro de baixo mostra exatamente os 28×28 bytes que vão para a placa.
4. **Inferir**. Depois de cada inferência pelo app, o firmware segura o resultado
   no LCD até **KEY1**; pressione-o antes de inferir de novo. Os pesos não são
   reenviados entre imagens.

Feche o app antes de usar a coleta em lote: ele mantém a porta serial aberta.

## Coleta em lote (`eval_batch.py`)

Envia as N primeiras imagens do conjunto de teste à placa e grava, por imagem, a
classe, os logits, os ciclos de cada camada e o contador de falhas da SDRAM, e
compara tudo com a simulação bit-exata do firmware feita no host (a partir do
`weights.bin` e do `model_meta.h` do pacote). Usa o comando `H` do firmware
(inferência sem esperar KEY1) e, a cada `--sw-every` imagens, o comando `S`, que
repete a inferência só na CPU para medir o ganho do acelerador.

```bash
.venv/Scripts/python eval_batch.py --port COM10 --model models/mnist \
    --data ../inference_tcc/data/mnist --n 200 --sw-every 50 --load --out mnist.csv
```

`--load` grava firmware e pesos antes (pede KEY0); sem ele, a placa já precisa
estar com o modelo. Saídas: o CSV, `<out>.log` (texto bruto da UART) e um resumo
no terminal. As coletas do TCC estão em `docs/dados/` do repositório do texto.

## Pacotes de modelo

Cada modelo é uma pasta em `models/`:

| Arquivo | Conteúdo |
|---|---|
| `model.json` | nome, classes, entrada, pré-processamento, amostras |
| `firmware.bin` | `neorv32_exe.bin` compilado com o `model_meta.h` do modelo |
| `weights.bin` | pesos INT8 + bias INT32 |
| `model_meta.h` | cópia para referência |
| `samples/*.bin` | imagens de teste, 784 B INT8 |

O firmware faz parte do pacote porque as escalas e zero-points do
`model_meta.h` são compilados nele: trocar só os pesos daria resultados errados.

Os pacotes de `models/` contêm o firmware usado nas medições do TCC, compilado do
commit `d800e48c` do ramo `systolic-array-accelerator` do `neorv32` (então em
`sw/example/neorv32_mnist_infer`): recompilar esse commit com o `model_meta.h` de cada
modelo reproduz os `firmware.bin` byte a byte.
Os `weights.bin` e `model_meta.h` são idênticos aos de `inference_tcc/out/<dataset>/`.

Para gerar ou atualizar um pacote a partir de `inference_tcc/out/<dataset>`,
com o toolchain RISC-V (`riscv-none-elf-gcc`) e `make` no PATH:

```bash
export NEORV32_HOME=<caminho do repositório neorv32>
.venv/Scripts/python make_package.py ../inference_tcc/out/mnist
.venv/Scripts/python make_package.py ../inference_tcc/out/fashion_mnist
```

O firmware é compilado numa cópia temporária de `sw/tcc/mnist_infer`, e o
repositório neorv32 não é alterado. Um pacote gerado assim usa o firmware do
estado atual do ramo, que pode ser posterior ao das medições. Para um dataset
novo, passe `--classes a,b,c,...` e `--preprocess` (`mnist`: dígito centrado em
20×20; `fill`: objeto ocupa 28×28).

## Arquivos

- `app.py`: interface (PySide6)
- `board.py`: protocolo serial (bootloader + handshake `R`/`W`/`I` e comandos `G`/`H`/`S`), sem Qt
- `eval_batch.py`: coleta em lote com referência bit-exata
- `imaging.py`: conversão de imagens para o tensor INT8
- `packages.py`: leitura dos pacotes em `models/`
- `make_package.py`: gera pacotes
