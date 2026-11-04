# -*- coding: utf-8 -*-
"""
Generates the illustrated tutorial (PDF, Portuguese) for the Arduino/ESP32
streaming test -- how to set it up and run it.

It draws two diagrams with matplotlib (the PC <-> board loop and the anatomy of
an episode) and lays out the document with reportlab, reusing the styles of
make_arduino_hil_pdf.py.

Usage:
    python examples/make_arduino_stream_pdf.py                # both documents
    python examples/make_arduino_stream_pdf.py --doc rapido   # quick guide only
    python examples/make_arduino_stream_pdf.py --out-dir docs

Output (default out-dir: examples):
    <out-dir>/tutorial_stream_arduino_pt.pdf  full tutorial
    <out-dir>/guia_rapido_arduino_pt.pdf      quick guide: wiring, built-in
                                              datasets, before vs after run
Requirements: matplotlib, reportlab (pip install reportlab).

The content documents examples/arduino_stream_test.py -- keep both in sync when
the test script changes.
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image,  # noqa: E402
                                KeepTogether, PageTemplate, Spacer)

from make_arduino_hil_pdf import (AZUL, AZUL_CLARO, CINZA, CINZA_RL, CODE, LARANJA,  # noqa: E402
                                  LARANJA_CLARO, LI, TABELA, VERDE, VERDE_CLARO, P,
                                  _caixa, _seta)

VERMELHO = '#b71c1c'

# Helvetica/Courier (the built-in PDF fonts) have no arrow glyph: arrows use the
# DejaVu Sans that ships with matplotlib.
pdfmetrics.registerFont(TTFont('DejaVu', os.path.join(matplotlib.get_data_path(),
                                                      'fonts', 'ttf', 'DejaVuSans.ttf')))
MONO = '<font face="Courier">{}</font>'


def M(txt):
    """Inline monospace text."""
    return MONO.format(txt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


# ==============================================================================
# DIAGRAMS
# ==============================================================================

def figura_fluxo(path):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.text(5, 4.78, 'O PC faz o papel dos sensores; a placa classifica e dispara o alarme',
            ha='center', fontsize=12, weight='bold', color=AZUL)

    _caixa(ax, 0.10, 2.05, 3.25, 2.20,
           'PC\n(arduino_stream_test.py run)\n\nlê scenario.csv\nenvia 1 amostra a cada\n'
           '--period-ms e espera a resposta', AZUL, AZUL_CLARO, fontsize=9)
    _caixa(ax, 6.65, 2.05, 3.25, 2.20,
           'Arduino / ESP32\n(sketch before ou after)\n\npyra_predict() de produção\n'
           'conta classes anormais\nseguidas -> alarme + LED', LARANJA, LARANJA_CLARO, fontsize=9)

    _seta(ax, (3.45, 3.75), (6.57, 3.75), AZUL)
    ax.text(5.01, 3.86, 'D,<seq>,f0,...,fN-1*CS', ha='center', va='bottom', fontsize=8.5,
            color=AZUL, family='DejaVu Sans Mono')
    _seta(ax, (6.57, 2.95), (3.45, 2.95), LARANJA)
    ax.text(5.01, 2.62, 'C,<seq>,<classe>,<us>,<streak>,<alarme>', ha='center',
            va='bottom', fontsize=8.0, color=LARANJA, family='DejaVu Sans Mono')
    ax.text(5.01, 2.20, 'H = handshake   X = zera o alarme', ha='center', fontsize=8,
            color=CINZA, family='DejaVu Sans Mono')

    _caixa(ax, 1.70, 0.15, 6.60, 1.25,
           'comparação com o oráculo (calculado no prepare)\n'
           'classe por amostra  +  alarme por episódio  +  latência  +  memória',
           VERDE, VERDE_CLARO, fontsize=9.3)
    _seta(ax, (1.72, 2.05), (3.0, 1.40), VERDE)
    _seta(ax, (8.28, 2.05), (7.0, 1.40), VERDE)

    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def figura_episodio(path):
    """One schematic episode: 12 normal rows, onset, confirm=3, alarm."""
    n_norm, n_anom, confirm = 12, 10, 3
    # classes the board returns: 0 = normal, 1 = anomaly. One isolated blip
    # before the onset (streak 1, no alarm) and a detection 3 samples after it.
    cls = [0] * n_norm + [1] * n_anom
    cls[5] = 1
    cls[n_norm] = 0  # the first anomalous row is still classified normal
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    ax.set_xlim(-0.8, n_norm + n_anom + 0.3)
    ax.set_ylim(-1.9, 3.2)
    ax.axis('off')
    ax.text((n_norm + n_anom) / 2 - 0.5, 3.0, 'Anatomia de um episódio (--confirm 3)',
            ha='center', fontsize=12, weight='bold', color=AZUL)

    ax.add_patch(Rectangle((-0.5, 1.75), n_norm, 0.55, facecolor=VERDE_CLARO,
                           edgecolor=VERDE))
    ax.text(n_norm / 2 - 0.5, 2.02, 'linhas normais (--normal-rows)', ha='center',
            va='center', fontsize=9)
    ax.add_patch(Rectangle((n_norm - 0.5, 1.75), n_anom, 0.55, facecolor=LARANJA_CLARO,
                           edgecolor=LARANJA))
    ax.text(n_norm + n_anom / 2 - 0.5, 2.02, 'linhas da anomalia (--anomaly-rows)',
            ha='center', va='center', fontsize=9)

    ax.text(-0.75, 0.95, 'classe', ha='right', va='center', fontsize=8.5, color=CINZA)
    ax.text(-0.75, 0.15, 'streak', ha='right', va='center', fontsize=8.5, color=CINZA)
    streak, alarm_at = 0, None
    for i, c in enumerate(cls):
        streak = streak + 1 if c else 0
        cor = LARANJA if c else VERDE
        ax.add_patch(plt.Circle((i, 0.95), 0.32, color=cor, alpha=0.9))
        ax.text(i, 0.95, 'A' if c else 'N', ha='center', va='center', fontsize=8,
                color='white', weight='bold')
        ax.text(i, 0.15, str(streak), ha='center', va='center', fontsize=8.5,
                family='DejaVu Sans Mono')
        if i >= n_norm and streak >= confirm and alarm_at is None:
            alarm_at = i
            break
    for i in range(alarm_at + 1, len(cls)):
        ax.add_patch(plt.Circle((i, 0.95), 0.32, facecolor='white', edgecolor='#bbbbbb',
                                linestyle='--'))

    ax.annotate('início (onset)', xy=(n_norm - 0.5, 1.72), xytext=(n_norm - 0.5, -1.15),
                ha='center', fontsize=8.5, color=LARANJA,
                arrowprops=dict(arrowstyle='-', color=LARANJA, linestyle='--'))
    ax.annotate('ALARME: 3 anormais seguidas\n(LED acende, episódio para)',
                xy=(alarm_at, 0.6), xytext=(alarm_at + 2.6, -1.25), ha='center',
                fontsize=8.5, color=VERMELHO, weight='bold',
                arrowprops=dict(arrowstyle='-|>', color=VERMELHO))
    ax.annotate('erro isolado: streak 1,\nnão chega a disparar', xy=(5, 0.6),
                xytext=(5, -1.25), ha='center', fontsize=8.5, color=CINZA,
                arrowprops=dict(arrowstyle='-|>', color=CINZA))
    ax.text(alarm_at + 3.5, 0.15, 'não enviadas', ha='center', va='center',
            fontsize=8, color='#888888')

    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ==============================================================================
# CONTENT
# ==============================================================================

def conteudo(larg, fig1, fig2):
    E = [Spacer(1, 4),
         P('Teste de streaming em Arduino/ESP32<br/>antes × depois do refinamento', 'titulo'),
         P('Como configurar e executar ' + M('examples/arduino_stream_test.py') +
           ' — do pacote de teste no seu PC até o relatório gerado com a placa física',
           'subtitulo')]

    # ---------------------------------------------------------------- 1
    E += [P('1. O que este teste mede', 'h1'),
          P('O pyruleanalyzer refina as regras de um modelo de árvores (Decision Tree, Random '
            'Forest ou GBDT) e exporta o resultado para um sketch Arduino. Este teste responde, '
            '<b>na placa real</b>, se o modelo refinado continua detectando anomalias tão bem '
            'quanto o original — e quanto ele economiza em memória e tempo de inferência.'),
          P('O PC faz o papel dos sensores: envia <b>linhas reais do conjunto de teste</b> '
            '(nunca usadas no treino nem no refinamento), uma a cada ' + M('--period-ms') +
            '. O Arduino classifica cada amostra e dispara um <b>alarme</b> (e acende o '
            + M('LED_BUILTIN') + ') depois de ' + M('--confirm') + ' classificações anormais '
            'seguidas. O mesmo cenário é tocado duas vezes: com o sketch <b>before</b> (regras '
            'iniciais) e com o sketch <b>after</b> (regras finais).'),
          Image(fig1, width=larg, height=larg * 4.6 / 9.2),
          P('Figura 1 — o laço PC / placa. Como a lógica do alarme é determinística, o '
            + M('prepare') + ' já sabe em qual amostra cada variante deve alarmar (o '
            '“oráculo”); o ' + M('run') + ' confere a placa contra ele.', 'legenda')]

    # ---------------------------------------------------------------- 2
    E += [P('2. Três comandos, duas pessoas', 'h1'),
          P('O script é dividido para que quem tem a placa precise apenas de uma pasta (o '
            '<b>pacote</b>), sem a base de dados de treino nem o pyruleanalyzer instalado:'),
          TABELA([
              ['Comando', 'Quem roda', 'O que faz'],
              [(M('prepare'), 'celula'), 'Você, no seu PC (sem placa)',
               'Treina ou carrega o modelo, exporta os dois sketches, monta o cenário com '
               'linhas reais de teste, calcula o oráculo e escreve o pacote.'],
              [(M('run'), 'celula'), 'Quem tem a placa',
               'Grava cada sketch, toca o cenário, registra cada amostra e escreve '
               + M('results/<execução>/') + '.'],
              [(M('report'), 'celula'), 'Qualquer um',
               'Reimprime a comparação before × after de uma execução (o ' + M('run') +
               ' já faz isso no final).'],
          ], [larg * 0.15, larg * 0.27, larg * 0.58])]

    # ---------------------------------------------------------------- 3
    E += [P('3. Configuração (uma vez só)', 'h1'),
          P('3.1 No PC que prepara o pacote', 'h2'),
          P('Precisa do projeto completo (o ' + M('prepare') + ' treina com scikit-learn). Da '
            'raiz do repositório:'),
          CODE('pip install -e .                     # pyruleanalyzer + sklearn/pandas/numpy'),
          P('3.2 No PC ligado à placa', 'h2'),
          P('Basta Python 3.9+ e a pasta ' + M('examples/') + ' do repositório (no mínimo '
            + M('arduino_stream_test.py') + ' e ' + M('arduino_hardware_test.py') + ', que '
            'ele importa):'),
          CODE('pip install numpy pandas pyserial matplotlib\n'
               '# matplotlib é opcional: só gera o gráfico comparison.png'),
          P('Para gravar a placa há duas opções:'),
          LI('<b>arduino-cli</b> (automático, recomendado): o script compila, mede a Flash/SRAM '
             'reais e grava cada variante sozinho.'),
          LI('<b>Arduino IDE</b> (manual): com ' + M('--manual-upload') + ' o script pausa e '
             'pede para você gravar cada sketch e apertar Enter. Não há medição de memória.'),
          CODE('# Windows (ou baixe em https://arduino.github.io/arduino-cli/)\n'
               'winget install ArduinoSA.CLI\n'
               '\n'
               'arduino-cli core update-index\n'
               'arduino-cli core install arduino:avr     # Uno / Nano / Mega / Leonardo\n'
               'arduino-cli core install esp32:esp32     # ESP32 (requer a URL do board\n'
               '                                         # manager da Espressif)\n'
               'arduino-cli board list                   # confere a placa e a porta'),
          P('<b>Drivers:</b> clones de Nano e placas ESP32 costumam usar os chips CH340 ou '
            'CP2102; no Windows instale o driver correspondente se a porta não aparecer.',
            'nota'),
          P('3.3 Placas suportadas', 'h2'),
          P('O ' + M('--board') + ' do ' + M('prepare') + ' define o FQBN gravado no pacote e os '
            'limites de memória usados na estimativa. No ' + M('run') + ' ele pode ser trocado '
            'por ' + M('--board') + ' ou ' + M('--fqbn') + '.'),
          TABELA([
              ['--board', 'FQBN', 'Observação'],
              [(M('uno'), 'celula'), (M('arduino:avr:uno'), 'celula'),
               '2 KB de SRAM: bom para modelos pequenos'],
              [(M('nano'), 'celula'), (M('arduino:avr:nano:cpu=atmega328'), 'celula'),
               'igual ao Uno'],
              [(M('nano_old'), 'celula'), (M('arduino:avr:nano:cpu=atmega328old'), 'celula'),
               'Nano com bootloader antigo (upload falha com ' + M('nano') + ')'],
              [(M('mega'), 'celula'), (M('arduino:avr:mega'), 'celula'),
               '8 KB de SRAM: recomendado para bases com muitas features'],
              [(M('leonardo'), 'celula'), (M('arduino:avr:leonardo'), 'celula'),
               'USB nativo (o sketch espera a serial abrir)'],
              [(M('esp32'), 'celula'), (M('esp32:esp32:esp32'), 'celula'),
               'double real de 8 bytes, folga de memória'],
          ], [larg * 0.15, larg * 0.42, larg * 0.43]),
          P('Cada linha ' + M('D,...') + ' é guardada inteira num buffer na SRAM. Com muitas '
            'features (linhas acima de ~600 bytes) o ' + M('prepare') + ' avisa: prefira um '
            'Mega ou um ESP32.', 'nota')]

    # ---------------------------------------------------------------- 4
    E += [CondPageBreak(6 * cm),
          P('4. Passo 1 — preparar o pacote (' + M('prepare') + ')', 'h1'),
          P('Teste rápido com uma base embutida (iris, wine, breast_cancer ou digits):'),
          CODE('python examples/arduino_stream_test.py prepare \\\n'
               '    --dataset iris --board uno --name iris_dt'),
          P('Base real com registro temporal (os episódios seguem as transições '
            'normal <font name="DejaVu">→</font> anomalia de verdade da gravação):'),
          CODE('python examples/arduino_stream_test.py prepare \\\n'
               '    --train-csv data/swat/train.csv --test-csv data/swat/test.csv \\\n'
               '    --target "Normal/Attack" --time-col Timestamp --drop-cols Timestamp \\\n'
               '    --normal-class Normal --model "Random Forest" --n-estimators 10 \\\n'
               '    --max-depth 6 --board mega --name swat_rf'),
          P('Reaproveitando um modelo já treinado e refinado (um ' + M('PyRuleAnalyzer') +
            ' salvo com pickle). As colunas dos CSVs precisam ser as mesmas do treino:'),
          CODE('python examples/arduino_stream_test.py prepare --model-pkl files/model.pkl \\\n'
               '    --train-csv ... --test-csv ... --target ... --name meu_modelo'),
          P('4.1 Opções principais', 'h2'),
          TABELA([
              ['Opção', 'Padrão', 'Para que serve'],
              [(M('--dataset'), 'celula'), 'iris', 'Base embutida, quando nenhum CSV é dado.'],
              [(M('--csv'), 'celula'), '—', 'Um único CSV, dividido 70/30.'],
              [(M('--train-csv / --test-csv'), 'celula'), '—',
               'Divisão própria. O cenário usa <b>só</b> as linhas de teste.'],
              [(M('--target'), 'celula'), 'última coluna', 'Coluna do rótulo.'],
              [(M('--drop-cols'), 'celula'), '—', 'Colunas a ignorar, separadas por vírgula.'],
              [(M('--time-col'), 'celula'), '—',
               'Coluna de tempo do teste: episódios cortados nas transições reais. Sem ela, '
               'linhas normais são emendadas a linhas de uma classe anômala (em rodízio).'],
              [(M('--model'), 'celula'), 'Decision Tree',
               'Decision Tree, Random Forest ou Gradient Boosting Decision Trees.'],
              [(M('--n-estimators / --max-depth'), 'celula'), '— / 6',
               'Tamanho do modelo (impacta direto Flash e tempo).'],
              [(M('--model-pkl'), 'celula'), '—', 'Modelo pronto em vez de treinar.'],
              [(M('--normal-class'), 'celula'), 'a mais frequente',
               'Rótulo (ou código) da operação normal.'],
              [(M('--episodes'), 'celula'), '20', 'Quantos episódios no cenário.'],
              [(M('--normal-rows / --anomaly-rows'), 'celula'), '30 / 30',
               'Linhas antes e a partir do início da anomalia.'],
              [(M('--confirm'), 'celula'), '3', 'Anormais seguidas que disparam o alarme.'],
              [(M('--period-ms'), 'celula'), '100', 'Uma amostra a cada N ms.'],
              [(M('--board / --baud'), 'celula'), 'uno / 115200', 'Placa alvo e velocidade.'],
              [(M('--output-dir / --name'), 'celula'), 'files/stream / stream_model',
               'Onde o pacote é escrito: ' + M('<output-dir>/<name>') + '.'],
          ], [larg * 0.33, larg * 0.19, larg * 0.48]),
          P('4.2 O que sai', 'h2'),
          P('No final o ' + M('prepare') + ' imprime o comportamento esperado de cada variante, '
            'por exemplo:'),
          CODE('[5/5] Expected behaviour on the board (oracle)\n'
               '  before: detects 6/6 episodes, 0 false alarms, mean delay 3.00 samples\n'
               '  after : detects 6/6 episodes, 0 false alarms, mean delay 3.00 samples\n'
               '\n'
               '  package: files/stream/iris_dt\n'
               '  send this folder to whoever has the board (LEIAME.md).'),
          CODE('files/stream/iris_dt/\n'
               '|-- LEIAME.md              instruções para quem tem a placa\n'
               '|-- manifest.json          modelo, classes, episódios e oráculo\n'
               '|-- scenario.csv           cada amostra a enviar + classe esperada\n'
               '`-- sketches/\n'
               '    |-- iris_dt_before/iris_dt_before.ino\n'
               '    `-- iris_dt_after/iris_dt_after.ino'),
          P('<b>Envie a pasta inteira.</b> Não edite os sketches nem o ' + M('scenario.csv') +
            ': o handshake confere o id do modelo gravado na placa e o teste aborta se o sketch '
            'errado estiver nela.', 'nota')]

    # ---------------------------------------------------------------- 5
    E += [KeepTogether([
              P('5. Como o cenário funciona', 'h1'),
              P('Um episódio é “a planta funciona normalmente e então começa uma anomalia”. A '
                'placa mantém um <i>streak</i> de classificações não normais seguidas; quando ele '
                'chega a ' + M('--confirm') + ' o alarme dispara.'),
              Image(fig2, width=larg, height=larg * 3.5 / 9.2),
              P('Figura 2 — N = classe normal, A = qualquer outra classe. O episódio termina no '
                'alarme; aqui a primeira linha anômala ainda saiu normal, então o atraso de '
                'detecção é de 4 amostras.', 'legenda')]),
          LI('<b>Detecção:</b> o primeiro alarme a partir do início da anomalia encerra o '
             'episódio; o atraso é contado em amostras e em milissegundos.'),
          LI('<b>Falso alarme:</b> um alarme <i>antes</i> do início. É registrado, o alarme é '
             'zerado (comando ' + M('X') + ') e a fase normal continua.'),
          LI('<b>Perda (miss):</b> o episódio chega à última linha de anomalia sem alarme.')]

    # ---------------------------------------------------------------- 6
    E += [P('6. Passo 2 — ensaio sem placa (' + M('--port sim') + ')', 'h1'),
          P('Antes de mandar o pacote, rode-o no seu PC com uma placa virtual que responde a '
            'classe esperada. Isso testa o protocolo, o pacote e o script — <b>não</b> o código '
            'C (para isso use o host-check da seção 11).'),
          CODE('python examples/arduino_stream_test.py run \\\n'
               '    --package files/stream/iris_dt --port sim'),
          CODE('[1/2] Variant "before"\n'
               '  handshake: H,before,1ffef3a13c,4,3,0,0,3\n'
               '    episode   0: alarm at +3 samples\n'
               '    ...\n'
               '[2/2] Variant "after"\n'
               '  handshake: H,after,271d725b56,4,3,0,0,3\n'
               '    ...\n'
               '**Result: PASS** (every sample answered, 100% class parity, alarms where\n'
               'the oracle says)'),
          P('O relatório de um ensaio traz a marca <i>Loopback dry run</i>; tempos de '
            'inferência e memória aparecem como 0 ou n/a.')]

    # ---------------------------------------------------------------- 7
    E += [P('7. Passo 3 — rodar na placa (' + M('run') + ')', 'h1'),
          P('7.1 Antes de começar', 'h2'),
          LI('Conecte a placa e descubra a porta: ' + M('arduino-cli board list') +
             ' (ou Ferramentas <font name="DejaVu">→</font> Porta na IDE). No Windows é ' + M('COMx') + '; no Linux, '
             + M('/dev/ttyACM0') + ' ou ' + M('/dev/ttyUSB0') + '; no macOS, '
             + M('/dev/cu.usbmodem…') + '.'),
          LI('<b>Feche o Monitor Serial da IDE</b>: a porta é exclusiva e o upload/handshake '
             'falham se ela estiver aberta.'),
          LI('Rode da raiz do repositório (ou de onde o ' + M('examples/') + ' estiver).'),
          P('7.2 Com arduino-cli (automático)', 'h2'),
          CODE('python examples/arduino_stream_test.py run --package CAMINHO/iris_dt --port COM5'),
          P('Se ' + M('--port') + ' for omitido e o arduino-cli estiver instalado, o script '
            'tenta detectar a porta da placa sozinho.'),
          P('7.3 Com a Arduino IDE (manual)', 'h2'),
          CODE('python examples/arduino_stream_test.py run --package CAMINHO/iris_dt \\\n'
               '    --port COM5 --manual-upload'),
          P('O script mostra o caminho de cada ' + M('.ino') + ' (primeiro ' +
            M('sketches/<nome>_before') + ', depois ' + M('sketches/<nome>_after') + '). Abra-o '
            'na IDE, selecione a placa e a porta, grave, <b>feche o Monitor Serial</b> e aperte '
            'Enter no terminal.'),
          P('7.4 O que acontece em cada variante', 'h2'),
          LI('Compila e grava o sketch (lendo a Flash e a SRAM reais da saída do arduino-cli).'),
          LI('Abre a serial e espera ' + M('--boot-delay') + ' s (placas AVR reiniciam ao abrir '
             'a porta).'),
          LI('Handshake ' + M('H') + ': confere variante e id do modelo; aborta se não baterem.'),
          LI('Para cada episódio: ' + M('X') + ' (zera o alarme) e uma linha ' + M('D') + ' por '
             'tick, esperando a resposta ' + M('C') + '. Respostas inválidas são reenviadas até '
             + M('--retries') + ' vezes.'),
          P('7.5 Opções do run', 'h2'),
          TABELA([
              ['Opção', 'Padrão', 'Para que serve'],
              [(M('--package'), 'celula'), 'obrigatório', 'Pasta gerada pelo prepare.'],
              [(M('--port'), 'celula'), 'auto', M('COM5') + ', ' + M('/dev/ttyACM0') + ' ou '
               + M('sim') + '.'],
              [(M('--board / --fqbn'), 'celula'), 'do pacote', 'Troca a placa alvo.'],
              [(M('--baud'), 'celula'), 'do pacote', 'Precisa bater com o sketch.'],
              [(M('--arduino-cli'), 'celula'), 'arduino-cli', 'Caminho do executável.'],
              [(M('--manual-upload'), 'celula'), '—', 'Pausa para gravar pela IDE.'],
              [(M('--skip-upload'), 'celula'), '—',
               'Não grava; use junto com ' + M('--variants') + ' para repetir uma variante que '
               'já está na placa.'],
              [(M('--variants'), 'celula'), 'before after', 'Quais variantes rodar.'],
              [(M('--period-ms'), 'celula'), 'do pacote', 'Muda a taxa de amostragem.'],
              [(M('--boot-delay'), 'celula'), '2.5 s', 'Espera após abrir a porta.'],
              [(M('--timeout / --retries'), 'celula'), '3.0 s / 2',
               'Tempo máximo por resposta e reenvios.'],
          ], [larg * 0.28, larg * 0.19, larg * 0.53]),
          P('Exemplo: repetir só a variante <b>after</b>, que já está gravada:'),
          CODE('python examples/arduino_stream_test.py run --package CAMINHO/iris_dt \\\n'
               '    --port COM5 --skip-upload --variants after')]

    # ---------------------------------------------------------------- 8
    E += [CondPageBreak(7 * cm),
          P('8. Passo 4 — resultados e relatório', 'h1'),
          P('Cada execução cria ' + M('results/<placa>_<data>_<hora>/') + ' dentro do pacote:'),
          TABELA([
              ['Arquivo', 'Conteúdo'],
              [(M('before_samples.csv'), 'celula'),
               'Cada amostra enviada: classe esperada, classe da placa, rótulo real, µs de '
               'inferência, streak, alarme, RTT e perda de prazo.'],
              [(M('after_samples.csv'), 'celula'), 'Idem para a variante after.'],
              [(M('before.json / after.json'), 'celula'),
               'Métricas, resultado de cada episódio e erros de comunicação.'],
              [(M('comparison.md / .json'), 'celula'), 'Tabela before × after e o veredito.'],
              [(M('comparison.png'), 'celula'),
               'Atraso de detecção por episódio (perdas marcadas com x).'],
              [(M('run.json'), 'celula'), 'Placa, FQBN, porta, host e taxa usados.'],
          ], [larg * 0.30, larg * 0.70]),
          P('<b>Quem rodou na placa deve devolver a pasta ' + M('results/') + ' inteira.</b> '
            'Para reimprimir a comparação (da última execução, ou de uma específica):'),
          CODE('python examples/arduino_stream_test.py report --package CAMINHO/iris_dt\n'
               'python examples/arduino_stream_test.py report --package CAMINHO/iris_dt \\\n'
               '    --run uno_20261104_151206'),
          P('8.1 Lendo a tabela', 'h2'),
          TABELA([
              ['Métrica', 'Significado'],
              ['Rules', 'Número de regras do modelo (antes × depois do refinamento).'],
              ['Flash / SRAM measured', 'Uso real, lido do arduino-cli (n/a com upload manual).'],
              ['Inference mean / p95 (µs)', 'Tempo de ' + M('pyra_predict()') + ' medido com '
               + M('micros()') + ' na placa.'],
              ['Class parity board vs Python', 'Amostras em que a placa deu a mesma classe que o '
               'Python. Deve ser 100%.'],
              ['Accuracy vs true label', 'Acerto contra o rótulo real da linha.'],
              ['Episodes detected / delay', 'Detecções e atraso médio/máximo (amostras e ms).'],
              ['False alarms (/1000 normal)', 'Alarmes antes do início da anomalia.'],
              ['Alarm parity with oracle', 'Episódios em que a placa alarmou exatamente onde o '
               'oráculo previa.'],
              ['Deadline misses', 'Respostas que chegaram depois do próximo tick: inferência '
               'mais lenta que a taxa dos sensores.'],
              ['Communication errors', 'Timeouts, checksum ruim, respostas inválidas.'],
          ], [larg * 0.34, larg * 0.66]),
          P('<b>Critério de aprovação (PASS):</b> para cada variante, (1) todas as amostras '
            'respondidas, (2) 100% de paridade de classe e (3) alarmes em todos os episódios '
            'exatamente onde o oráculo diz. O processo sai com código 0 (PASS) ou 1 (FAIL), '
            'pronto para uso em CI. Acurácia, detecção e falsos alarmes são os números a '
            '<i>comparar</i> entre before e after, não critérios de aprovação.', 'nota')]

    # ---------------------------------------------------------------- 9
    E += [P('9. Protocolo serial', 'h1'),
          P('Texto, uma mensagem por linha terminada em ' + M('\\n') + '. O PC só envia a '
            'próxima amostra depois da resposta, então o buffer de 64 bytes do Uno nunca '
            'transborda.'),
          TABELA([
              ['Direção', 'Mensagem'],
              ['PC <font name="DejaVu">→</font> placa', (M('H'), 'celula')],
              ['placa <font name="DejaVu">→</font> PC', (M('H,<variante>,<model_id>,<n_feat>,<n_classes>,<n_trees>,'
                                '<normal>,<confirm>'), 'celula')],
              ['PC <font name="DejaVu">→</font> placa', (M('X') + ' &nbsp;(zera alarme e streak) <font name="DejaVu">→</font> ' + M('X,ok'), 'celula')],
              ['PC <font name="DejaVu">→</font> placa', (M('D,<seq>,f0,...,fN-1*CS') + ' &nbsp;(CS = XOR dos bytes antes '
                                'do *, 2 dígitos hex)', 'celula')],
              ['placa <font name="DejaVu">→</font> PC', (M('C,<seq>,<classe>,<inferencia_us>,<streak>,<alarme 0/1>'),
                              'celula')],
              ['placa <font name="DejaVu">→</font> PC', (M('E,<motivo>') + ' &nbsp;(checksum ruim, nº de campos errado, '
                                'linha longa demais…)', 'celula')],
          ], [larg * 0.17, larg * 0.83]),
          P('Ao ligar, o sketch imprime ' + M('PYRA-STREAM-READY') + '. Isso permite testar à mão '
            'no Monitor Serial (115200, fim de linha “Nova linha”): envie ' + M('H') + ' e veja '
            'a resposta — mas feche o monitor antes de rodar o script.')]

    # ---------------------------------------------------------------- 10
    E += [P('10. Problemas comuns', 'h1'),
          TABELA([
              ['Mensagem / sintoma', 'Causa provável e solução'],
              [(M('No handshake from the board'), 'celula'),
               'Porta errada, Monitor Serial aberto, baud diferente ou sketch de streaming não '
               'gravado. Confira com ' + M('arduino-cli board list') + '. Em placas lentas para '
               'reiniciar, aumente ' + M('--boot-delay') + '.'],
              [(M('The board runs "before" model ..., expected "after"'), 'celula'),
               'O sketch errado está na placa (comum com ' + M('--manual-upload') + ' ou '
               + M('--skip-upload') + '). Grave ' + M('sketches/<nome>_<variante>') + '.'],
              [(M('arduino-cli not found'), 'celula'),
               'Instale o arduino-cli, passe ' + M('--arduino-cli C:\\caminho\\arduino-cli.exe')
               + ' ou use ' + M('--manual-upload') + '.'],
              [(M('Compilation ... failed'), 'celula'),
               'Normalmente o modelo não cabe: reduza ' + M('--max-depth') + ' / '
               + M('--n-estimators') + ' no prepare ou use Mega/ESP32. Confirme que o core da '
               'placa está instalado.'],
              [(M('Upload failed'), 'celula'),
               'Monitor Serial aberto, porta errada ou Nano antigo (use ' + M('--board nano_old')
               + ').'],
              [(M('Pass --port ...'), 'celula'),
               'A porta não foi detectada: informe ' + M('--port') + ' explicitamente.'],
              ['Paridade de classe abaixo de 100%',
               'Em AVR, ' + M('double') + ' tem 4 bytes: amostras quase exatamente sobre um '
               'limiar podem divergir. Se todas as divergências estão na fronteira, não é bug de '
               'lógica; no ESP32 o efeito some. Rode o host-check (seção 11) para separar '
               'exportador × placa.'],
              ['Muitos deadline misses',
               'A inferência é mais lenta que ' + M('--period-ms') + ': aumente o período ou '
               'diminua o modelo. Não invalida o teste, mas mostra que a placa não acompanha '
               'aquela taxa.'],
              [(M('E,line_too_long') + ' / ' + M('E,bad_feature_count'), 'celula'),
               'Muitas features para o buffer da placa, ou CSV com colunas diferentes do '
               'treino. Use Mega/ESP32 e confira ' + M('--drop-cols') + '.'],
          ], [larg * 0.36, larg * 0.64])]

    # ---------------------------------------------------------------- 11
    E += [P('11. Os outros testes com Arduino do repositório', 'h1'),
          P('O teste de streaming fecha um ciclo de três scripts em ' + M('examples/') + '. Vale '
            'rodar os outros dois antes, porque eles isolam problemas mais cedo:'),
          TABELA([
              ['Script', 'Precisa de placa?', 'O que verifica'],
              [(M('arduino_hardware_test.py'), 'celula'), 'Opcional',
               'Teste HIL vetor a vetor: paridade C × Python, latência e memória. Com '
               + M('--host-check --no-hardware') + ' compila o mesmo C com g++ no PC (ideal '
               'para CI). Tutorial próprio: ' + M('tutorial_hil_arduino_pt.pdf') + '.'],
              [(M('arduino_sensor_simulation.py'), 'celula'), 'Não',
               'Roda o sketch de produção (setup/loop) no PC com g++ e um shim de Arduino, '
               'alimentado por sinal sintético ou pelas linhas reais (ADC ou serial).'],
              [(M('arduino_stream_test.py'), 'celula'), 'Sim (ou ' + M('sim') + ')',
               'Este tutorial: streaming na placa real, before × after, com alarme.'],
          ], [larg * 0.33, larg * 0.17, larg * 0.50]),
          CODE('# só no PC, sem placa: valida o C gerado contra o Python\n'
               'python examples/arduino_hardware_test.py --dataset iris --host-check --no-hardware\n'
               '\n'
               '# simulação contínua do sketch de produção no PC\n'
               'python examples/arduino_sensor_simulation.py'),
          P('Ordem sugerida: host-check <font name="DejaVu">→</font> simulação <font name="DejaVu">→</font> ' + M('prepare') + ' <font name="DejaVu">→</font> ' + M('run --port sim')
            + ' <font name="DejaVu">→</font> ' + M('run') + ' na placa <font name="DejaVu">→</font> ' + M('report') + '.')]

    return E


# ==============================================================================
# CONTENT -- quick guide (practical steps only)
# ==============================================================================

SETA = '<font name="DejaVu">→</font>'


def conteudo_rapido(larg, fig1, fig2):
    E = [Spacer(1, 4),
         P('Guia rápido: testar o modelo no Arduino', 'titulo'),
         P('Só os passos práticos: ligar a placa, escolher um dataset que já vem pronto e '
           'comparar, na placa, o modelo <b>antes</b> e <b>depois</b> do refinamento',
           'subtitulo')]

    # ---------------------------------------------------------------- 0
    E += [P('O que você precisa', 'h1'),
          LI('Uma placa: <b>Arduino Uno, Nano, Mega</b> ou <b>ESP32</b>.'),
          LI('Um cabo USB <b>de dados</b> (alguns cabos só carregam e a placa não aparece no '
             'PC).'),
          LI('Um PC com Python 3.9+ e este repositório baixado.'),
          LI('Nada mais: <b>não precisa ligar sensores</b>. O PC envia os dados pelo USB e a '
             'placa responde. O LED da placa (pino 13 / ' + M('LED_BUILTIN') + ') acende quando '
             'ela detecta uma anomalia.')]

    # ---------------------------------------------------------------- 1
    E += [P('Passo 1: instalar (só na primeira vez)', 'h1'),
          P('Abra um terminal na pasta do repositório e rode:'),
          CODE('pip install -e .\n'
               'pip install pyserial matplotlib\n'
               '\n'
               'winget install ArduinoSA.CLI            # Windows; nos outros sistemas veja\n'
               '                                        # arduino.github.io/arduino-cli\n'
               'arduino-cli core update-index\n'
               'arduino-cli core install arduino:avr    # Uno, Nano, Mega\n'
               'arduino-cli core install esp32:esp32    # só se for usar ESP32'),
          P('Feche e abra o terminal de novo depois do ' + M('winget') + ', para o comando '
            + M('arduino-cli') + ' ser encontrado. Prefere a Arduino IDE? Pule o arduino-cli e '
            'veja a dica no fim do Passo 4.', 'nota')]

    # ---------------------------------------------------------------- 2
    E += [P('Passo 2: ligar o Arduino e achar a porta', 'h1'),
          LI('Conecte a placa no USB. O LED de energia (ON / PWR) deve acender.'),
          LI('Descubra a porta:'),
          CODE('arduino-cli board list'),
          CODE('Port   Protocol Type              Board Name  FQBN\n'
               'COM5   serial   Serial Port (USB) Arduino Uno arduino:avr:uno'),
          P('Anote a porta (aqui, ' + M('COM5') + '). No Linux ela aparece como '
            + M('/dev/ttyACM0') + ' ou ' + M('/dev/ttyUSB0') + '; no macOS, '
            + M('/dev/cu.usbmodem...') + '.'),
          P('<b>A placa não aparece?</b> Troque o cabo (pode ser só de carga); em clones de '
            'Nano e em ESP32 instale o driver <b>CH340</b> ou <b>CP2102</b> (veja o nome do '
            'chip ao lado do USB); no Windows, confira em Gerenciador de Dispositivos ' + SETA +
            ' Portas (COM e LPT); no Linux, rode ' + M('sudo usermod -aG dialout $USER') +
            ' e entre de novo na sessão.', 'nota'),
          P('<b>Feche o Monitor Serial da Arduino IDE</b> se ele estiver aberto: só um '
            'programa pode usar a porta por vez.', 'nota')]

    # ---------------------------------------------------------------- 3
    E += [CondPageBreak(7 * cm),
          P('Passo 3: escolher o dataset', 'h1'),
          P('Os quatro datasets abaixo já vêm com o scikit-learn: não precisa baixar nada. Os '
            'números são de uma execução real com as opções padrão.'),
          TABELA([
              ['--dataset', 'Features', 'Regras antes ' + SETA + ' depois', 'Placa'],
              [(M('iris'), 'celula'), '4', '8 ' + SETA + ' 4',
               'qualquer uma (comece por este)'],
              [(M('wine'), 'celula'), '13', '6 ' + SETA + ' 4', 'qualquer uma'],
              [(M('breast_cancer'), 'celula'), '30', '16 ' + SETA + ' 7', 'qualquer uma'],
              [(M('digits'), 'celula'), '64', '38 ' + SETA + ' 26',
               '<b>Mega ou ESP32</b> (no Uno/Nano a linha de 64 valores ocupa mais da metade '
               'dos 2 KB de RAM)'],
          ], [larg * 0.20, larg * 0.13, larg * 0.24, larg * 0.43]),
          P('No comando do passo seguinte, troque ' + M('--board') + ' pela sua placa: '
            + M('uno') + ', ' + M('nano') + ' (ou ' + M('nano_old') + ' para Nanos com '
            'bootloader antigo), ' + M('mega') + ', ' + M('leonardo') + ' ou ' + M('esp32')
            + '.')]

    # ---------------------------------------------------------------- 4
    E += [P('Passo 4: rodar (antes e depois do refinamento)', 'h1'),
          P('O teste sempre compara <b>duas versões do mesmo modelo</b>, cada uma gravada e '
            'testada na placa:'),
          TABELA([
              ['Versão', 'O que é', 'Sketch gravado'],
              [(M('before'), 'celula'), 'Modelo original, com as regras iniciais.',
               (M('sketches/iris_before/'), 'celula')],
              [(M('after'), 'celula'), 'Modelo depois do refinamento (regras redundantes '
               'removidas).', (M('sketches/iris_after/'), 'celula')],
          ], [larg * 0.13, larg * 0.50, larg * 0.37]),
          P('As duas recebem exatamente as mesmas amostras. Os comandos abaixo usam o '
            + M('iris') + ' num Uno na ' + M('COM5') + '; troque pelo seu dataset, placa e '
            'porta.'),
          P('4a. Gerar o pacote de teste (cria as duas versões)', 'h2'),
          CODE('python examples/arduino_stream_test.py prepare --dataset iris --board uno '
               '--name iris'),
          P('Treina o modelo, refina as regras e cria a pasta ' + M('files/stream/iris/') + ' '
            'com os dois sketches. No final ele já mostra o que espera de cada versão:'),
          CODE('  rules: before 8 -> after 4\n'
               '  ...\n'
               '  before: detects 20/20 episodes, 0 false alarms, mean delay 3.00 samples\n'
               '  after : detects 20/20 episodes, 0 false alarms, mean delay 3.00 samples'),
          P('4b. Ensaio sem placa (opcional, 2 minutos)', 'h2'),
          CODE('python examples/arduino_stream_test.py run --package files/stream/iris '
               '--port sim'),
          P('Uma placa de mentira responde no lugar do Arduino. Serve só para confirmar que '
            'o pacote está certo antes de mexer no hardware.'),
          P('4c. Rodar na placa', 'h2'),
          CODE('python examples/arduino_stream_test.py run --package files/stream/iris '
               '--port COM5'),
          P('O script faz tudo sozinho, em ordem:'),
          LI('<b>[1/2] before:</b> compila e grava o sketch ' + M('iris_before') + ', envia as '
             'amostras e registra as respostas.'),
          LI('<b>[2/2] after:</b> grava o sketch ' + M('iris_after') + ' por cima e repete '
             'exatamente o mesmo cenário.'),
          LI('No fim imprime a tabela <b>Before × After</b> e o resultado.'),
          P('Com o ' + M('iris') + ' cada versão leva cerca de 1 minuto. Durante o teste o LED '
            'da placa pisca: ele acende a cada anomalia detectada e apaga quando o próximo '
            'trecho normal começa.'),
          P('<b>Sem arduino-cli (usando a Arduino IDE):</b> acrescente '
            + M('--manual-upload') + '. O script para duas vezes e mostra qual ' + M('.ino')
            + ' gravar (primeiro o before, depois o after). Grave pela IDE, feche o Monitor '
            'Serial e aperte Enter.', 'nota'),
          P('4d. Rodar só uma das versões', 'h2'),
          P('Para repetir só o <b>after</b> (ou só o <b>before</b>), use ' + M('--variants')
            + '. Se o sketch certo já estiver gravado na placa, ' + M('--skip-upload') + ' '
            'pula a gravação:'),
          CODE('python examples/arduino_stream_test.py run --package files/stream/iris \\\n'
               '    --port COM5 --variants after\n'
               '\n'
               'python examples/arduino_stream_test.py run --package files/stream/iris \\\n'
               '    --port COM5 --variants after --skip-upload'),
          P('No Windows (PowerShell/cmd), escreva cada comando numa linha só, sem a '
            + M('\\') + '.', 'nota')]

    # ---------------------------------------------------------------- 5
    E += [CondPageBreak(8 * cm),
          P('Passo 5: ler o resultado (antes × depois)', 'h1'),
          P('A tabela final coloca as duas versões lado a lado. As linhas que mais importam:'),
          TABELA([
              ['Linha da tabela', 'O que comparar entre Before e After'],
              ['Rules', 'Quantas regras cada versão tem (o after deve ter menos).'],
              ['Flash / SRAM measured', 'Memória usada na placa (o after deve usar menos).'],
              ['Inference mean (us)', 'Tempo de cada classificação, em microssegundos.'],
              ['Accuracy vs true label', 'Acerto contra o rótulo real.'],
              ['Episodes detected / False alarms', 'Se o refinamento manteve a detecção de '
               'anomalias sem gerar alarmes falsos.'],
              ['Class parity board vs Python', 'Tem que dar <b>100%</b> nas duas: a placa '
               'classifica igual ao Python.'],
          ], [larg * 0.38, larg * 0.62]),
          P('A última linha diz <b>Result: PASS</b> ou <b>FAIL</b>. PASS quer dizer que, nas '
            'duas versões, a placa respondeu tudo, classificou igual ao Python e alarmou '
            'onde devia.'),
          P('Os arquivos ficam em ' + M('files/stream/iris/results/<placa>_<data>/') + ': '
            + M('comparison.md') + ' (a tabela), ' + M('comparison.png') + ' (gráfico) e os '
            'CSVs com cada amostra de ' + M('before') + ' e ' + M('after') + '. Para ver a '
            'tabela de novo:'),
          CODE('python examples/arduino_stream_test.py report --package files/stream/iris')]

    # ---------------------------------------------------------------- 6
    E += [CondPageBreak(9 * cm),
          P('Se der errado', 'h1'),
          TABELA([
              ['O que aparece', 'O que fazer'],
              [(M('No handshake from the board'), 'celula'),
               'Porta errada ou Monitor Serial aberto. Confira ' + M('arduino-cli board list')
               + ', feche a IDE e rode de novo.'],
              [(M('arduino-cli not found'), 'celula'),
               'Reabra o terminal depois de instalar, ou use ' + M('--manual-upload') + '.'],
              [(M('Upload failed'), 'celula'),
               'Monitor Serial aberto ou porta errada. Nano: tente ' + M('--board nano_old')
               + '. ESP32: segure o botão <b>BOOT</b> quando aparecer “Connecting...”.'],
              [(M('Compilation ... failed'), 'celula'),
               'O core da placa não está instalado (Passo 1) ou o modelo não cabe: use um '
               'dataset menor ou um Mega/ESP32.'],
              [(M('The board runs "before"...'), 'celula'),
               'O sketch errado está na placa: rode sem ' + M('--skip-upload') + '.'],
              ['Parity 99,x%', 'Normal em Uno/Nano com valores exatamente no limite de uma '
               'regra (a placa tem menos precisão decimal). No ESP32 não acontece.'],
          ], [larg * 0.36, larg * 0.64]),
          P('Detalhes de cada opção, do protocolo e das métricas estão no tutorial completo, '
            + M('tutorial_stream_arduino_pt.pdf') + '.')]

    return E


# ==============================================================================
# BUILD
# ==============================================================================

DOCS = {
    'completo': {'file': 'tutorial_stream_arduino_pt.pdf', 'content': conteudo,
                 'title': 'Teste de streaming em Arduino/ESP32 — pyruleanalyzer'},
    'rapido': {'file': 'guia_rapido_arduino_pt.pdf', 'content': conteudo_rapido,
               'title': 'Guia rápido: testar o modelo no Arduino — pyruleanalyzer'},
}


def gerar(out_dir, work_dir, qual='completo'):
    fig1 = os.path.join(work_dir, 'stream_fluxo.png')
    fig2 = os.path.join(work_dir, 'stream_episodio.png')
    figura_fluxo(fig1)
    figura_episodio(fig2)

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, DOCS[qual]['file'])
    doc = BaseDocTemplate(pdf_path, pagesize=A4,
                          leftMargin=2 * cm, rightMargin=2 * cm,
                          topMargin=1.8 * cm, bottomMargin=2 * cm,
                          title=DOCS[qual]['title'],
                          author='pyruleanalyzer')

    def rodape(canvas, doc_):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(CINZA_RL)
        canvas.drawString(2 * cm, 1.2 * cm, 'pyruleanalyzer — teste de streaming em Arduino/ESP32')
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f'página {doc_.page}')
        canvas.setStrokeColor(colors.HexColor('#c9d3de'))
        canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
        canvas.restoreState()

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    doc.addPageTemplates([PageTemplate(id='padrao', frames=[frame], onPage=rodape)])
    doc.build(DOCS[qual]['content'](doc.width, fig1, fig2))
    print(f'  {pdf_path} ({os.path.getsize(pdf_path):,} bytes)')
    return pdf_path


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)),
                   help='output directory (default: the examples folder)')
    p.add_argument('--doc', default='both', choices=['completo', 'rapido', 'both'],
                   help='full tutorial, quick guide, or both (default)')
    args = p.parse_args()
    work = tempfile.mkdtemp(prefix='pyra_stream_pdf_')
    print('Generating the streaming tutorial PDF(s):')
    for qual in (['completo', 'rapido'] if args.doc == 'both' else [args.doc]):
        gerar(args.out_dir, work, qual)


if __name__ == '__main__':
    main()
