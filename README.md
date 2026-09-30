# NEORV32 Inferência (desktop)

App desktop para rodar modelos CNN INT8 no NEORV32 + acelerador sistólico na
DE2-115: escolhe o modelo, grava na placa, escolhe uma imagem e mostra a classe
prevista, o tempo e a pontuação por classe.

Conversa com o firmware `sw/example/neorv32_mnist_infer` do repositório neorv32
pela mesma UART do `send_weights.py`. Os modelos vêm do repositório
`inference_tcc`.

## Instalação

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Uso

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
4. **Inferir**. Depois de cada inferência o firmware segura o resultado no LCD
   até **KEY1**; pressione-o antes ou depois de clicar em *Inferir* de novo.
   Os pesos não são reenviados entre imagens.

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

Para gerar ou atualizar um pacote a partir de `inference_tcc/out/<dataset>`,
com o toolchain RISC-V (`riscv-none-elf-gcc`) e `make` no PATH:

```bash
export NEORV32_HOME=C:/Users/vinic/OneDrive/Documentos/univasf/neorv32
.venv/Scripts/python make_package.py ../inference_tcc/out/mnist
.venv/Scripts/python make_package.py ../inference_tcc/out/fashion_mnist
```

O firmware é compilado numa cópia temporária, e o repositório neorv32 não é
alterado. Para um dataset novo, passe `--classes a,b,c,...` e `--preprocess`
(`mnist`: dígito centrado em 20×20; `fill`: objeto ocupa 28×28).

## Arquivos

- `app.py`: interface (PySide6)
- `board.py`: protocolo serial (bootloader + handshake `R`/`W`/`I`/`G`), sem Qt
- `imaging.py`: conversão de imagens para o tensor INT8
- `packages.py`: leitura dos pacotes em `models/`
- `make_package.py`: gera pacotes
