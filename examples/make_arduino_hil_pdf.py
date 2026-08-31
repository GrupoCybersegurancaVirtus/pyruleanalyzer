# -*- coding: utf-8 -*-
"""
Generates the illustrated tutorial (PDF) for the Arduino/ESP32 HIL test pipeline.

It builds two diagrams with matplotlib (the data flow and the 7 stages) and lays
out the document with reportlab, in Portuguese and/or English.

Usage:
    python examples/make_arduino_hil_pdf.py              # both languages
    python examples/make_arduino_hil_pdf.py --lang pt    # Portuguese only
    python examples/make_arduino_hil_pdf.py --lang en    # English only
    python examples/make_arduino_hil_pdf.py --out-dir docs

Output: <out-dir>/tutorial_hil_arduino_pt.pdf and _en.pdf (default out-dir: examples).
Requirements: matplotlib, reportlab (pip install reportlab).

The content documents examples/arduino_hardware_test.py — keep both in sync when
the test script changes.
"""

import argparse
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

AZUL = '#1f4e79'
AZUL_CLARO = '#dce9f5'
LARANJA = '#c55a11'
LARANJA_CLARO = '#fce4d6'
VERDE = '#2e7d32'
VERDE_CLARO = '#e3f3e4'
CINZA = '#555555'


# ==============================================================================
# DIAGRAMS
# ==============================================================================

FIG_TXT = {
    'pt': {
        'titulo1': 'Como o teste funciona: o PC vira a fonte dos dados',
        'pc': 'PC\n(arduino_hardware_test.py)\n\n'
              'treina o modelo\ngera o sketch HIL\nprediz com o Python',
        'placa': 'Arduino / ESP32\n\n'
                 'mesmos arrays das árvores\nmesma pyra_predict()\nresponde pela serial',
        'resposta': 'R,<id>,<classe>,<microssegundos>,<reps>',
        'comparacao': 'comparação classe a classe\n'
                      'paridade C × Python  +  latência  +  memória',
        'titulo2': 'As 7 etapas do pipeline',
        'etapas': [
            ('1', 'Treino + exportação',
             'full_pipeline(generate_arduino_sketch=True)\n'
             'separa o holdout e gera o .ino de produção', 'host'),
            ('2', 'Geração do sketch HIL',
             'recorta dados + motor do .ino, corrige defeitos\n'
             'e cola o harness serial por cima', 'host'),
            ('3', 'Host-check com g++',
             'compila o mesmo C no PC e compara com o Python\n'
             '--host-check   (esta é a etapa de CI)', 'host'),
            ('4', 'Compilação',
             'arduino-cli compile --fqbn <fqbn> <pasta>\n'
             'lê o uso REAL de Flash e SRAM', 'hw'),
            ('5', 'Upload',
             'arduino-cli upload -p <porta> --fqbn <fqbn> <pasta>', 'hw'),
            ('6', 'Teste na placa (HIL)',
             'handshake "I", envia N vetores, coleta a classe\n'
             'e o tempo de inferência de cada um', 'hw'),
            ('7', 'Relatório + gate',
             'JSON + TXT, exit code 0 (passou) ou 1 (falhou)', 'fim'),
        ],
        'legenda': ['roda só no PC', 'precisa da placa', 'resultado / aprovação'],
    },
    'en': {
        'titulo1': 'How the test works: the PC becomes the data source',
        'pc': 'PC\n(arduino_hardware_test.py)\n\n'
              'trains the model\nbuilds the HIL sketch\npredicts with Python',
        'placa': 'Arduino / ESP32\n\n'
                 'same tree arrays\nsame pyra_predict()\nanswers over serial',
        'resposta': 'R,<id>,<class>,<microseconds>,<reps>',
        'comparacao': 'class-by-class comparison\n'
                      'C × Python parity  +  latency  +  memory',
        'titulo2': 'The 7 stages of the pipeline',
        'etapas': [
            ('1', 'Training + export',
             'full_pipeline(generate_arduino_sketch=True)\n'
             'holds out a test set and emits the production .ino', 'host'),
            ('2', 'HIL sketch generation',
             'cuts data + engine from the .ino, patches defects\n'
             'and appends the serial harness', 'host'),
            ('3', 'Host-check with g++',
             'compiles the same C on the PC and compares to Python\n'
             '--host-check   (this is the CI stage)', 'host'),
            ('4', 'Compilation',
             'arduino-cli compile --fqbn <fqbn> <folder>\n'
             'reads the REAL Flash and SRAM usage', 'hw'),
            ('5', 'Upload',
             'arduino-cli upload -p <port> --fqbn <fqbn> <folder>', 'hw'),
            ('6', 'Test on the board (HIL)',
             '"I" handshake, sends N vectors, collects the class\n'
             'and the inference time of each one', 'hw'),
            ('7', 'Report + gate',
             'JSON + TXT, exit code 0 (pass) or 1 (fail)', 'fim'),
        ],
        'legenda': ['PC only', 'needs the board', 'result / pass criteria'],
    },
}


def _caixa(ax, x, y, w, h, texto, cor_borda, cor_fundo, fontsize=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.08',
                                linewidth=1.6, edgecolor=cor_borda, facecolor=cor_fundo))
    ax.text(x + w / 2, y + h / 2, texto, ha='center', va='center',
            fontsize=fontsize, color='#1a1a1a', linespacing=1.5)


def _seta(ax, p1, p2, cor=CINZA):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=14,
                                 linewidth=1.4, color=cor, shrinkA=2, shrinkB=2))


def figura_fluxo(path, lang):
    t = FIG_TXT[lang]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    ax.text(5, 4.25, t['titulo1'], ha='center', fontsize=12.5, weight='bold', color=AZUL)

    _caixa(ax, 0.15, 1.95, 3.15, 1.65, t['pc'], AZUL, AZUL_CLARO, fontsize=9.5)
    _caixa(ax, 6.70, 1.95, 3.15, 1.65, t['placa'], LARANJA, LARANJA_CLARO, fontsize=9.5)

    _seta(ax, (3.40, 3.20), (6.62, 3.20), AZUL)
    ax.text(5.01, 3.32, 'P,<id>,<reps>,f0,f1,...,fN', ha='center', va='bottom',
            fontsize=8.5, color=AZUL, family='DejaVu Sans Mono')

    _seta(ax, (6.62, 2.55), (3.40, 2.55), LARANJA)
    ax.text(5.01, 2.28, t['resposta'], ha='center', va='bottom',
            fontsize=8.5, color=LARANJA, family='DejaVu Sans Mono')

    _caixa(ax, 2.60, 0.30, 4.80, 1.05, t['comparacao'], VERDE, VERDE_CLARO, fontsize=9.5)
    _seta(ax, (1.72, 1.95), (3.55, 1.35), VERDE)
    _seta(ax, (8.28, 1.95), (6.45, 1.35), VERDE)

    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def figura_etapas(path, lang):
    t = FIG_TXT[lang]
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.2)
    ax.axis('off')

    ax.text(5, 8.92, t['titulo2'], ha='center', fontsize=12.5, weight='bold', color=AZUL)

    y, alt, gap = 8.45, 0.88, 0.18
    for num, titulo, desc, tipo in t['etapas']:
        borda, fundo = {'host': (AZUL, AZUL_CLARO),
                        'hw': (LARANJA, LARANJA_CLARO),
                        'fim': (VERDE, VERDE_CLARO)}[tipo]

        ax.add_patch(FancyBboxPatch((1.15, y - alt), 7.9, alt,
                                    boxstyle='round,pad=0.02,rounding_size=0.06',
                                    linewidth=1.5, edgecolor=borda, facecolor=fundo))
        ax.add_patch(plt.Circle((0.72, y - alt / 2), 0.27, color=borda, zorder=3))
        ax.text(0.72, y - alt / 2, num, ha='center', va='center', color='white',
                fontsize=11, weight='bold', zorder=4)
        ax.text(1.42, y - 0.25, titulo, ha='left', va='center', fontsize=10,
                weight='bold', color='#1a1a1a')
        ax.text(1.42, y - alt + 0.29, desc, ha='left', va='center', fontsize=8.2,
                color='#333333', family='DejaVu Sans Mono', linespacing=1.35)

        if num != '7':
            ax.add_patch(FancyArrowPatch((5.1, y - alt), (5.1, y - alt - gap + 0.02),
                                         arrowstyle='-|>', mutation_scale=12,
                                         linewidth=1.2, color=CINZA))
        y -= alt + gap

    larguras = [2.30, 2.30, 2.80]
    cores = [(AZUL, AZUL_CLARO), (LARANJA, LARANJA_CLARO), (VERDE, VERDE_CLARO)]
    x = 1.15
    for w, (borda, fundo), txt in zip(larguras, cores, t['legenda']):
        ax.add_patch(FancyBboxPatch((x, 0.10), w, 0.44,
                                    boxstyle='round,pad=0.02,rounding_size=0.06',
                                    linewidth=1.4, edgecolor=borda, facecolor=fundo))
        ax.text(x + w / 2, 0.32, txt, ha='center', va='center', fontsize=8.5)
        x += w + 0.25

    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ==============================================================================
# DOCUMENT STYLES
# ==============================================================================

AZUL_RL = colors.HexColor(AZUL)
CINZA_RL = colors.HexColor('#444444')

_base = getSampleStyleSheet()
S = {
    'titulo': ParagraphStyle('titulo', parent=_base['Title'], fontSize=21,
                             textColor=AZUL_RL, spaceAfter=6, leading=25),
    'subtitulo': ParagraphStyle('subtitulo', parent=_base['Normal'], fontSize=11,
                                textColor=CINZA_RL, alignment=1, spaceAfter=16,
                                leading=15),
    'h1': ParagraphStyle('h1', parent=_base['Heading1'], fontSize=15,
                         textColor=AZUL_RL, spaceBefore=15, spaceAfter=8, leading=19),
    'h2': ParagraphStyle('h2', parent=_base['Heading2'], fontSize=12,
                         textColor=colors.HexColor('#2b6ca3'), spaceBefore=11,
                         spaceAfter=5, leading=15),
    'p': ParagraphStyle('p', parent=_base['BodyText'], fontSize=10, leading=14.5,
                        alignment=TA_JUSTIFY, spaceAfter=7),
    'li': ParagraphStyle('li', parent=_base['BodyText'], fontSize=10, leading=14.5,
                         leftIndent=14, bulletIndent=4, spaceAfter=5,
                         alignment=TA_JUSTIFY),
    'code': ParagraphStyle('code', parent=_base['BodyText'], fontName='Courier',
                           fontSize=7.9, leading=11.2,
                           textColor=colors.HexColor('#1a1a1a'),
                           backColor=colors.HexColor('#f2f4f7'), borderPadding=6,
                           leftIndent=4, rightIndent=4, spaceBefore=4, spaceAfter=9),
    'legenda': ParagraphStyle('legenda', parent=_base['Normal'], fontSize=8.5,
                              textColor=CINZA_RL, alignment=1, spaceAfter=10,
                              spaceBefore=2),
    'nota': ParagraphStyle('nota', parent=_base['BodyText'], fontSize=9.4, leading=13.5,
                           leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=9,
                           backColor=colors.HexColor('#fff8e6'), borderPadding=6,
                           alignment=TA_JUSTIFY),
    'celula': ParagraphStyle('celula', parent=_base['BodyText'], fontSize=8.6,
                             leading=11.5, spaceAfter=0),
    'celula_code': ParagraphStyle('celula_code', parent=_base['BodyText'],
                                  fontName='Courier', fontSize=7.8, leading=10.8,
                                  spaceAfter=0),
}


def P(txt, estilo='p'):
    return Paragraph(txt, S[estilo])


def LI(txt):
    return Paragraph(txt, S['li'], bulletText='\u2022')


def CODE(txt):
    txt = (txt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
           .replace('\n', '<br/>').replace(' ', '&nbsp;'))
    return Paragraph(txt, S['code'])


def TABELA(dados, larguras):
    linhas = []
    for i, linha in enumerate(dados):
        nova = []
        for cel in linha:
            estilo = 'celula'
            if isinstance(cel, tuple):
                cel, estilo = cel
            if i == 0:
                cel, estilo = f'<font color="white"><b>{cel}</b></font>', 'celula'
            nova.append(Paragraph(cel, S[estilo]))
        linhas.append(nova)
    t = Table(linhas, colWidths=larguras, repeatRows=1)
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c9d3de')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
        ('BACKGROUND', (0, 0), (-1, 0), AZUL_RL),
    ]))
    return t


# ==============================================================================
# CONTENT — Portuguese
# ==============================================================================

def conteudo_pt(larg, fig1, fig2):
    E = [Spacer(1, 4),
         P('Testando modelos pyruleanalyzer<br/>em Arduino e ESP32 reais', 'titulo'),
         P('Guia do script <font face="Courier">examples/arduino_hardware_test.py</font> '
           '— do treino em Python até a validação na placa física', 'subtitulo')]

    E += [P('1. O que este script faz', 'h1'),
          P('O <b>pyruleanalyzer</b> já exporta um modelo treinado (Decision Tree, Random Forest ou '
            'GBDT) para um sketch Arduino autocontido, com as árvores convertidas em arrays '
            '<font face="Courier">const</font> e uma função <font face="Courier">pyra_predict()</font> '
            'em C. O que faltava era responder à pergunta seguinte: '
            '<b>a placa realmente classifica igual ao Python?</b>'),
          P('O script <font face="Courier">examples/arduino_hardware_test.py</font> responde isso com '
            'um teste <i>hardware-in-the-loop</i> (HIL): ele treina o modelo, gera um sketch de teste, '
            'compila, grava na placa, envia amostras reais pela porta serial e compara, uma a uma, a '
            'classe devolvida pelo microcontrolador com a classe prevista em Python — medindo de quebra '
            'a latência de cada inferência e o uso real de Flash e SRAM.'),
          P('O sketch de <b>produção</b> lê sensores dentro de <font face="Courier">read_features()</font>; '
            'por isso ele não pode ser testado sozinho — os dados vêm do mundo físico. O truque do teste '
            'é inverter o fluxo: o PC passa a ser a fonte dos dados, e o sketch de <b>teste</b> reaproveita '
            'exatamente os mesmos arrays e a mesma <font face="Courier">pyra_predict()</font>, trocando '
            'apenas a leitura dos sensores por um protocolo serial. Valida-se, portanto, o mesmo código '
            'que vai para o campo.'),
          Image(fig1, width=larg, height=larg * 4.2 / 9.2),
          P('Figura 1 — o PC envia o vetor de features; a placa devolve a classe e o tempo de '
            'inferência. O que sobra é comparar.', 'legenda')]

    E += [P('2. Antes de começar: pré-requisitos', 'h1'),
          P('Instale uma vez só, na máquina que vai rodar o teste:'),
          CODE('pip install pyserial                 # comunicação serial com a placa\n'
               '\n'
               '# arduino-cli (compilar e gravar) - https://arduino.github.io/arduino-cli/\n'
               'winget install ArduinoSA.CLI         # Windows\n'
               'arduino-cli core update-index\n'
               'arduino-cli core install arduino:avr # Uno / Nano / Mega / Leonardo\n'
               'arduino-cli core install esp32:esp32 # ESP32 (exige a URL do board manager)\n'
               '\n'
               'arduino-cli board list               # confirme que a placa aparece'),
          LI('<b>Windows:</b> clones de Nano e ESP32 precisam do driver CH340 ou CP2102.'),
          LI('<b>g++</b> (opcional, mas recomendado): habilita a etapa 3, que valida o C no PC sem '
             'placa nenhuma.'),
          LI('<b>Feche o Monitor Serial</b> da Arduino IDE antes de rodar: a porta é exclusiva.')]

    E += [P('3. O pipeline de teste, etapa por etapa', 'h1'),
          Image(fig2, width=larg * 0.93, height=larg * 0.93 * 7.0 / 9.2),
          P('Figura 2 — as etapas em azul rodam só no PC; as laranjas exigem a placa conectada.',
            'legenda'),

          P('Etapa 1 — Treino e exportação', 'h2'),
          P('O script carrega o dataset (embutido do scikit-learn ou o seu CSV), separa um conjunto de '
            'teste que <b>não</b> participa do treino e chama '
            '<font face="Courier">full_pipeline(generate_arduino_sketch=True)</font>. Saem daí o '
            '<font face="Courier">.ino</font> de produção e a estimativa de memória '
            '(<font face="Courier">memory_check</font>).'),

          P('Etapa 2 — Geração do sketch HIL', 'h2'),
          P('O script recorta do <font face="Courier">.ino</font> tudo o que vem antes do marcador '
            '<font face="Courier">ARDUINO SKETCH SECTION</font> — ou seja, os dados das árvores e o motor '
            'de inferência — e cola por cima o <i>harness</i> serial. O resultado fica em '
            '<font face="Courier">files/hil/&lt;nome&gt;_hil/&lt;nome&gt;_hil.ino</font> '
            '(o Arduino exige que a pasta tenha o mesmo nome do arquivo).'),
          P('<b>De quebra, esta etapa conserta dois defeitos do gerador:</b> (1) o bloco '
            '<font face="Courier">Feature order:</font> sai sem <font face="Courier">//</font> e não '
            'compila; (2) o sketch de produção declara <font face="Courier">float features[]</font> e '
            'chama <font face="Courier">pyra_predict((const double*)features)</font> — um cast inválido '
            'que lê lixo em qualquer plataforma onde <font face="Courier">float != double</font>. O '
            'harness declara o buffer como <font face="Courier">double</font> e não usa cast.', 'nota'),

          P('Etapa 3 — Host-check com g++ (sem hardware)', 'h2'),
          P('Antes de gastar tempo gravando, o mesmo código C é compilado no PC com '
            '<font face="Courier">g++</font> e alimentado com os vetores de teste. Se a paridade '
            'C × Python já falha aqui, o problema está no modelo ou no exportador — e não na placa. '
            'É esta a etapa que você roda em integração contínua.'),

          P('Etapa 4 — Compilação para a placa', 'h2'),
          P('O <font face="Courier">arduino-cli compile</font> devolve o uso <b>real</b> de memória; o '
            'script extrai as linhas <font face="Courier">Sketch uses ... bytes</font> e '
            '<font face="Courier">Global variables use ... bytes</font> e as coloca no relatório ao lado '
            'da estimativa. O número real ser maior que o estimado é normal — entra o runtime do Arduino. '
            'O que importa é a porcentagem final.'),

          P('Etapa 5 — Upload', 'h2'),
          P('Gravação via <font face="Courier">arduino-cli upload -p &lt;porta&gt;</font>. Se a placa já '
            'estiver com o sketch HIL, use <font face="Courier">--skip-upload</font>; para pular também a '
            'compilação, <font face="Courier">--skip-compile</font>.'),

          P('Etapa 6 — O teste na placa', 'h2'),
          P('O script abre a serial (abrir a porta reseta placas AVR, por isso ele espera '
            '<font face="Courier">--boot-delay</font> segundos), faz o handshake com o comando '
            '<font face="Courier">I</font> e então envia as amostras. Cada resposta traz a classe e o '
            'tempo total de <font face="Courier">--reps</font> inferências, para que a resolução de '
            '<font face="Courier">micros()</font> não domine a medida.'),

          P('Etapa 7 — Relatório e critério de aprovação', 'h2'),
          P('Sai um <font face="Courier">.json</font> e um <font face="Courier">.txt</font> em '
            '<font face="Courier">--report-dir</font>. O processo termina com <b>exit code 0</b> (passou) '
            'ou <b>1</b> (falhou), pronto para servir de gate em CI. Reprova se: a paridade ficar abaixo '
            'de <font face="Courier">--min-parity</font> (padrão 100%), houver qualquer timeout ou '
            'resposta inválida, o host-check falhar, ou Flash/SRAM estourarem os limites da placa.')]

    E += [P('4. Como executar', 'h1'),
          P('4.1 Sem placa nenhuma (comece por aqui)', 'h2'),
          P('Valida o pipeline inteiro até a etapa 3. É o comando ideal para a primeira vez e para '
            'rodar em CI:'),
          CODE('python examples/arduino_hardware_test.py --dataset iris \\\n'
               '    --host-check --no-hardware'),

          P('4.2 Ciclo completo em uma placa real', 'h2'),
          CODE('# Arduino Uno na COM5 (Linux/Mac: /dev/ttyUSB0 ou /dev/ttyACM0)\n'
               'python examples/arduino_hardware_test.py --dataset iris \\\n'
               '    --board uno --port COM5 --host-check'),
          P('Omitindo <font face="Courier">--port</font>, o script tenta descobrir a placa sozinho com '
            '<font face="Courier">arduino-cli board list</font>.'),

          P('4.3 Outros cenários úteis', 'h2'),
          CODE('# ESP32 com Random Forest, 200 amostras e 50 repetições por amostra\n'
               'python examples/arduino_hardware_test.py --dataset wine \\\n'
               '    --model "Random Forest" --n-estimators 10 --max-depth 6 \\\n'
               '    --board esp32 --port COM7 --n-samples 200 --reps 50\n'
               '\n'
               '# Seu próprio CSV, com a placa já gravada\n'
               'python examples/arduino_hardware_test.py --csv files/dataset_iris.csv \\\n'
               '    --target Target --board nano --port COM3 --skip-upload\n'
               '\n'
               '# Reaproveitando um modelo já treinado\n'
               'python examples/arduino_hardware_test.py --model-pkl files/final_model.pkl \\\n'
               '    --board mega --port COM4'),

          P('4.4 Principais opções', 'h2'),
          TABELA([
              ['Opção', 'Para que serve'],
              [('--dataset / --csv', 'celula_code'),
               'Dataset embutido (iris, wine, breast_cancer, digits) ou o seu CSV, com '
               '<font face="Courier">--target</font> definindo a coluna alvo.'],
              [('--model', 'celula_code'),
               '"Decision Tree" (padrão), "Random Forest" ou "Gradient Boosting Decision Trees".'],
              [('--n-estimators<br/>--max-depth', 'celula_code'),
               'Tamanho do modelo. Em Uno/Nano, mantenha a profundidade baixa e poucas árvores.'],
              [('--board / --fqbn', 'celula_code'),
               'uno, nano, nano_old, mega, leonardo, esp32 — ou um FQBN explícito.'],
              [('--port / --baud', 'celula_code'),
               'Porta serial e velocidade (padrão 115200). Sem <font face="Courier">--port</font>, '
               'tenta auto-detectar.'],
              [('--n-samples / --reps', 'celula_code'),
               'Quantas amostras enviar e quantas repetições por amostra na medição de latência.'],
              [('--host-check', 'celula_code'),
               'Liga a etapa 3 (compila o motor com g++ e compara com o Python).'],
              [('--no-hardware', 'celula_code'),
               'Para depois da etapa 3: não compila nem grava.'],
              [('--skip-compile<br/>--skip-upload', 'celula_code'),
               'Pula a compilação e/ou a gravação quando a placa já está pronta.'],
              [('--min-parity', 'celula_code'),
               'Paridade mínima para o teste passar (padrão 100%).'],
              [('--output-dir / --report-dir', 'celula_code'),
               'Onde ficam os artefatos gerados e os relatórios.'],
          ], [larg * 0.31, larg * 0.69])]

    E += [P('5. O protocolo serial', 'h1'),
          P('Texto puro, uma mensagem por linha, terminada em <font face="Courier">\\n</font>. Você pode '
            'conversar com a placa na mão pelo Monitor Serial da IDE — útil para depurar:'),
          TABELA([
              ['Sentido', 'Mensagem', 'Significado'],
              ['PC -&gt; placa', ('I', 'celula_code'), 'Pergunta os metadados do modelo.'],
              ['placa -&gt; PC',
               ('I,&lt;algoritmo&gt;,&lt;n_features&gt;,<br/>&lt;n_classes&gt;,&lt;n_trees&gt;',
                'celula_code'), 'Resposta do handshake.'],
              ['PC -&gt; placa',
               ('P,&lt;id&gt;,&lt;reps&gt;,&lt;f0&gt;,&lt;f1&gt;,...', 'celula_code'),
               'Pede a classificação de um vetor, repetida &lt;reps&gt; vezes.'],
              ['placa -&gt; PC',
               ('R,&lt;id&gt;,&lt;classe&gt;,<br/>&lt;microssegundos&gt;,&lt;reps&gt;', 'celula_code'),
               'Classe prevista e tempo total das repetições.'],
              ['placa -&gt; PC', ('E,&lt;motivo&gt;', 'celula_code'),
               'Erro de parsing ou comando desconhecido.'],
          ], [larg * 0.16, larg * 0.44, larg * 0.40]),
          P('Exemplo manual, com um modelo de 4 features: enviar '
            '<font face="Courier">P,0,1,5.1,3.5,1.4,0.2</font> deve devolver algo como '
            '<font face="Courier">R,0,0,180,1</font> — classe 0, em 180 microssegundos.')]

    E += [P('6. Lendo o relatório', 'h1'),
          P('O <font face="Courier">.txt</font> gerado tem esta cara:'),
          CODE('HIL REPORT - pyruleanalyzer on hardware\n'
               'Model           : Decision Tree (Decision Tree)\n'
               'Board / FQBN    : uno / arduino:avr:uno\n'
               'Samples         : 50 (answered: 50)\n'
               '\n'
               'C/Python parity   : 100.00%\n'
               'Accuracy (host)   : 93.33%\n'
               'Accuracy (board)  : 93.33%\n'
               'Host-check (g++)  : 100.00% C/Python parity\n'
               '\n'
               'Latency per inference: mean 212.4 us | p50 208.0 | p95 244.0 | max 260.0\n'
               'Measured memory: Flash 3412 B (10%) | SRAM 268 B (13%)\n'
               '\n'
               'RESULT: PASS'),
          TABELA([
              ['Campo', 'Como interpretar'],
              [('C/Python parity', 'celula_code'),
               'O número que importa: mede se a placa concorda com o Python amostra a amostra. Abaixo de '
               '100%, o relatório lista as divergências (host, placa e rótulo verdadeiro).'],
              [('Accuracy (host) × (board)', 'celula_code'),
               'Acurácia contra o rótulo verdadeiro. Se as duas batem, o embarcado não perdeu qualidade.'],
              [('Host-check (g++)', 'celula_code'),
               'Paridade do mesmo C compilado no PC. Se falha aqui, o problema não é a placa.'],
              [('Latency', 'celula_code'),
               'Tempo por inferência (média, p50, p95 e máximo), já dividido pelas repetições. Use para '
               'dimensionar o período de amostragem.'],
              [('Measured memory', 'celula_code'),
               'Flash e SRAM reais reportados pelo toolchain, ao lado da estimativa do pyruleanalyzer.'],
              [('RESULT', 'celula_code'), 'PASS = exit code 0; FAIL = exit code 1.'],
          ], [larg * 0.30, larg * 0.70])]

    E += [P('7. Armadilhas que você vai encontrar', 'h1'),
          LI('<b>No AVR, <font face="Courier">double</font> é igual a <font face="Courier">float</font></b> '
             '(4 bytes). Os limiares são exportados com 17 dígitos e perdem precisão ao serem lidos pelo '
             '<font face="Courier">atof()</font> da placa. Amostras que caem quase em cima de um limiar '
             'podem divergir do Python. Paridade de 99,x% com todas as divergências na fronteira é isso — '
             'e não um erro de lógica. No ESP32 (double de 8 bytes) o efeito some.'),
          LI('<b>A ordem das features é um contrato.</b> A placa não conhece nomes, só índices. O vetor '
             'enviado precisa seguir a mesma ordem do treino — e <font face="Courier">read_features()</font>, '
             'no sketch de produção, tem a mesma obrigação.'),
          LI('<b>Uno e Nano têm 2 KB de SRAM.</b> Cada feature custa 4 bytes no buffer, mais cerca de 14 '
             'bytes de texto na linha serial. Acima de ~40 features o Uno aperta; prefira Mega ou ESP32.'),
          LI('<b>Placas AVR resetam ao abrir a porta serial.</b> O script espera '
             '<font face="Courier">--boot-delay</font> segundos (padrão 2,5 s) e descarta o banner antes '
             'do handshake.'),
          LI('<b>ESP32 e o watchdog.</b> Muitas repetições numa única mensagem podem disparar o cão de '
             'guarda; mantenha <font face="Courier">--reps</font> em até ~200.'),
          LI('<b>Timeout.</b> Modelos grandes (Random Forest com dezenas de árvores) levam milissegundos '
             'por inferência em um Uno; multiplique isso por <font face="Courier">--reps</font> antes de '
             'culpar o <font face="Courier">--timeout</font>.')]

    E += [P('8. Problemas comuns', 'h1'),
          TABELA([
              ['Sintoma', 'Causa provável e solução'],
              ['<i>The board did not answer the "I" handshake</i>',
               'Porta errada, baud diferente, sketch HIL não gravado, ou o Monitor Serial da IDE segurando '
               'a porta. Confira com <font face="Courier">arduino-cli board list</font> e feche a IDE.'],
              [('arduino-cli not found', 'celula_code'),
               'Instale o arduino-cli ou passe '
               '<font face="Courier">--arduino-cli C:\\caminho\\arduino-cli.exe</font>. Para apenas gerar '
               'o sketch, use <font face="Courier">--no-hardware</font>.'],
              [('pyserial is not installed', 'celula_code'),
               '<font face="Courier">pip install pyserial</font> (o pacote se chama pyserial, mas o import '
               'é <font face="Courier">serial</font>).'],
              ['Compilação estoura a memória',
               'Reduza <font face="Courier">--max-depth</font> e <font face="Courier">--n-estimators</font>, '
               'ou troque de placa. A estimativa já aparece na etapa 1.'],
              ['Paridade abaixo de 100%',
               'Rode com <font face="Courier">--host-check</font>: se o g++ também divergir, o problema é o '
               'modelo/exportador; se só a placa divergir, quase sempre é a precisão de float no AVR.'],
              ['Muitos timeouts',
               'Aumente <font face="Courier">--timeout</font> ou diminua <font face="Courier">--reps</font>; '
               'modelos grandes em AVR são lentos.'],
          ], [larg * 0.32, larg * 0.68])]

    E += [P('9. Usando em integração contínua', 'h1'),
          P('Sem placa, rode apenas a etapa 3 — ela garante que o exportador continua gerando C válido e '
            'equivalente ao Python:'),
          CODE('- name: HIL host-check\n'
               '  run: |\n'
               '    pip install -e .\n'
               '    python examples/arduino_hardware_test.py --dataset iris \\\n'
               '        --host-check --no-hardware --min-parity 100'),
          P('Em um runner auto-hospedado com a placa física conectada, use o comando completo com '
            '<font face="Courier">--port</font> fixo. Nos dois casos o exit code já funciona como critério '
            'de aprovação: não é preciso interpretar a saída.')]

    E += [P('10. Próximo passo: dos vetores do PC para os sensores reais', 'h1'),
          P('Depois que o teste HIL passa, você sabe que o motor de inferência está correto na placa. O que '
            'resta é trocar a fonte dos dados: no sketch de <b>produção</b>, preencha '
            '<font face="Courier">read_features()</font> com as leituras dos seus sensores, '
            '<b>na mesma ordem de features usada no treino</b>:'),
          CODE('void read_features(void) {\n'
               '    features[0] = analogRead(A0) * (5.0 / 1023.0);  // ex.: tensão do sensor 1\n'
               '    features[1] = dht.readTemperature();            // ex.: temperatura\n'
               '    features[2] = dht.readHumidity();               // ex.: umidade\n'
               '    // ... uma linha por feature, na ordem do treino\n'
               '}'),
          P('Uma boa prática é manter os dois sketches à mão: o de produção para o campo e o HIL para '
            'revalidar o modelo sempre que ele for retreinado.')]

    return E


# ==============================================================================
# CONTENT — English
# ==============================================================================

def conteudo_en(larg, fig1, fig2):
    E = [Spacer(1, 4),
         P('Testing pyruleanalyzer models<br/>on real Arduino and ESP32 boards', 'titulo'),
         P('A guide to <font face="Courier">examples/arduino_hardware_test.py</font> '
           '— from training in Python to validation on the physical board', 'subtitulo')]

    E += [P('1. What this script does', 'h1'),
          P('<b>pyruleanalyzer</b> already exports a trained model (Decision Tree, Random Forest or GBDT) '
            'to a self-contained Arduino sketch, with the trees turned into '
            '<font face="Courier">const</font> arrays and a <font face="Courier">pyra_predict()</font> '
            'function in C. What was missing was the answer to the next question: '
            '<b>does the board really classify the same way Python does?</b>'),
          P('The script <font face="Courier">examples/arduino_hardware_test.py</font> answers that with a '
            '<i>hardware-in-the-loop</i> (HIL) test: it trains the model, generates a test sketch, compiles '
            'it, flashes the board, sends real samples over the serial port and compares, one by one, the '
            'class returned by the microcontroller against the class predicted in Python — while also '
            'measuring the latency of each inference and the real Flash and SRAM usage.'),
          P('The <b>production</b> sketch reads sensors inside <font face="Courier">read_features()</font>; '
            'that is why it cannot be tested on its own — the data comes from the physical world. The trick '
            'is to invert the flow: the PC becomes the data source, and the <b>test</b> sketch reuses '
            'exactly the same arrays and the same <font face="Courier">pyra_predict()</font>, replacing only '
            'the sensor reading with a serial protocol. What gets validated is therefore the very code that '
            'ships to the field.'),
          Image(fig1, width=larg, height=larg * 4.2 / 9.2),
          P('Figure 1 — the PC sends the feature vector; the board returns the class and the inference '
            'time. All that is left is comparing them.', 'legenda')]

    E += [P('2. Before you start: prerequisites', 'h1'),
          P('Install once, on the machine that will run the test:'),
          CODE('pip install pyserial                 # serial communication with the board\n'
               '\n'
               '# arduino-cli (compile and flash) - https://arduino.github.io/arduino-cli/\n'
               'winget install ArduinoSA.CLI         # Windows\n'
               'arduino-cli core update-index\n'
               'arduino-cli core install arduino:avr # Uno / Nano / Mega / Leonardo\n'
               'arduino-cli core install esp32:esp32 # ESP32 (needs the board manager URL)\n'
               '\n'
               'arduino-cli board list               # confirm the board shows up'),
          LI('<b>Windows:</b> Nano and ESP32 clones need the CH340 or CP2102 driver.'),
          LI('<b>g++</b> (optional but recommended): enables stage 3, which validates the C code on the PC '
             'with no board at all.'),
          LI('<b>Close the Arduino IDE Serial Monitor</b> before running: the port is exclusive.')]

    E += [P('3. The test pipeline, stage by stage', 'h1'),
          Image(fig2, width=larg * 0.93, height=larg * 0.93 * 7.0 / 9.2),
          P('Figure 2 — the blue stages run on the PC alone; the orange ones need the board attached.',
            'legenda'),

          P('Stage 1 — Training and export', 'h2'),
          P('The script loads the dataset (a scikit-learn built-in or your own CSV), holds out a test set '
            'that does <b>not</b> take part in training, and calls '
            '<font face="Courier">full_pipeline(generate_arduino_sketch=True)</font>. Out of it come the '
            'production <font face="Courier">.ino</font> and the memory estimate '
            '(<font face="Courier">memory_check</font>).'),

          P('Stage 2 — HIL sketch generation', 'h2'),
          P('The script cuts from the <font face="Courier">.ino</font> everything before the '
            '<font face="Courier">ARDUINO SKETCH SECTION</font> marker — that is, the tree data and the '
            'inference engine — and appends the serial <i>harness</i> on top. The result lands in '
            '<font face="Courier">files/hil/&lt;name&gt;_hil/&lt;name&gt;_hil.ino</font> '
            '(Arduino requires the folder to carry the same name as the file).'),
          P('<b>Along the way, this stage fixes two defects of the generator:</b> (1) the '
            '<font face="Courier">Feature order:</font> block is emitted without '
            '<font face="Courier">//</font> and does not compile; (2) the production sketch declares '
            '<font face="Courier">float features[]</font> and calls '
            '<font face="Courier">pyra_predict((const double*)features)</font> — an invalid cast that reads '
            'garbage on any platform where <font face="Courier">float != double</font>. The harness declares '
            'the buffer as <font face="Courier">double</font> and uses no cast.', 'nota'),

          P('Stage 3 — Host-check with g++ (no hardware)', 'h2'),
          P('Before spending time flashing, the same C code is compiled on the PC with '
            '<font face="Courier">g++</font> and fed the test vectors. If C × Python parity already fails '
            'here, the problem is in the model or the exporter — not on the board. This is the stage you '
            'run in continuous integration.'),

          P('Stage 4 — Compiling for the board', 'h2'),
          P('<font face="Courier">arduino-cli compile</font> reports the <b>real</b> memory usage; the '
            'script extracts the <font face="Courier">Sketch uses ... bytes</font> and '
            '<font face="Courier">Global variables use ... bytes</font> lines and puts them in the report '
            'next to the estimate. The real number being larger than the estimate is expected — the Arduino '
            'runtime is included. What matters is the final percentage.'),

          P('Stage 5 — Upload', 'h2'),
          P('Flashing via <font face="Courier">arduino-cli upload -p &lt;port&gt;</font>. If the board '
            'already runs the HIL sketch, use <font face="Courier">--skip-upload</font>; to skip compiling '
            'as well, <font face="Courier">--skip-compile</font>.'),

          P('Stage 6 — The test on the board', 'h2'),
          P('The script opens the serial port (opening it resets AVR boards, hence the '
            '<font face="Courier">--boot-delay</font> wait), handshakes with the '
            '<font face="Courier">I</font> command and then sends the samples. Each response carries the '
            'class and the total time of <font face="Courier">--reps</font> inferences, so that the '
            'resolution of <font face="Courier">micros()</font> does not dominate the measurement.'),

          P('Stage 7 — Report and pass criteria', 'h2'),
          P('A <font face="Courier">.json</font> and a <font face="Courier">.txt</font> are written to '
            '<font face="Courier">--report-dir</font>. The process exits with <b>code 0</b> (pass) or '
            '<b>1</b> (fail), ready to be used as a CI gate. It fails when: parity drops below '
            '<font face="Courier">--min-parity</font> (default 100%), any timeout or invalid response '
            'shows up, the host-check fails, or Flash/SRAM exceed the limits of the board.')]

    E += [P('4. How to run it', 'h1'),
          P('4.1 With no board at all (start here)', 'h2'),
          P('Validates the whole pipeline up to stage 3. This is the command for your first run and for CI:'),
          CODE('python examples/arduino_hardware_test.py --dataset iris \\\n'
               '    --host-check --no-hardware'),

          P('4.2 Full cycle on a real board', 'h2'),
          CODE('# Arduino Uno on COM5 (Linux/Mac: /dev/ttyUSB0 or /dev/ttyACM0)\n'
               'python examples/arduino_hardware_test.py --dataset iris \\\n'
               '    --board uno --port COM5 --host-check'),
          P('If you omit <font face="Courier">--port</font>, the script tries to find the board itself with '
            '<font face="Courier">arduino-cli board list</font>.'),

          P('4.3 Other useful scenarios', 'h2'),
          CODE('# ESP32 with Random Forest, 200 samples and 50 repetitions per sample\n'
               'python examples/arduino_hardware_test.py --dataset wine \\\n'
               '    --model "Random Forest" --n-estimators 10 --max-depth 6 \\\n'
               '    --board esp32 --port COM7 --n-samples 200 --reps 50\n'
               '\n'
               '# Your own CSV, with the board already flashed\n'
               'python examples/arduino_hardware_test.py --csv files/dataset_iris.csv \\\n'
               '    --target Target --board nano --port COM3 --skip-upload\n'
               '\n'
               '# Reusing an already trained model\n'
               'python examples/arduino_hardware_test.py --model-pkl files/final_model.pkl \\\n'
               '    --board mega --port COM4'),

          P('4.4 Main options', 'h2'),
          TABELA([
              ['Option', 'What it does'],
              [('--dataset / --csv', 'celula_code'),
               'Built-in dataset (iris, wine, breast_cancer, digits) or your own CSV, with '
               '<font face="Courier">--target</font> naming the label column.'],
              [('--model', 'celula_code'),
               '"Decision Tree" (default), "Random Forest" or "Gradient Boosting Decision Trees".'],
              [('--n-estimators<br/>--max-depth', 'celula_code'),
               'Model size. On Uno/Nano, keep the depth low and the number of trees small.'],
              [('--board / --fqbn', 'celula_code'),
               'uno, nano, nano_old, mega, leonardo, esp32 — or an explicit FQBN.'],
              [('--port / --baud', 'celula_code'),
               'Serial port and speed (default 115200). Without <font face="Courier">--port</font>, it '
               'tries to auto-detect.'],
              [('--n-samples / --reps', 'celula_code'),
               'How many samples to send and how many repetitions per sample when measuring latency.'],
              [('--host-check', 'celula_code'),
               'Enables stage 3 (compiles the engine with g++ and compares against Python).'],
              [('--no-hardware', 'celula_code'),
               'Stops after stage 3: no compiling, no flashing.'],
              [('--skip-compile<br/>--skip-upload', 'celula_code'),
               'Skips compiling and/or flashing when the board is already set up.'],
              [('--min-parity', 'celula_code'),
               'Minimum parity for the test to pass (default 100%).'],
              [('--output-dir / --report-dir', 'celula_code'),
               'Where the generated artifacts and the reports go.'],
          ], [larg * 0.31, larg * 0.69])]

    E += [P('5. The serial protocol', 'h1'),
          P('Plain text, one message per line, terminated by <font face="Courier">\\n</font>. You can talk '
            'to the board by hand through the IDE Serial Monitor — handy for debugging:'),
          TABELA([
              ['Direction', 'Message', 'Meaning'],
              ['PC -&gt; board', ('I', 'celula_code'), 'Asks for the model metadata.'],
              ['board -&gt; PC',
               ('I,&lt;algorithm&gt;,&lt;n_features&gt;,<br/>&lt;n_classes&gt;,&lt;n_trees&gt;',
                'celula_code'), 'Handshake response.'],
              ['PC -&gt; board',
               ('P,&lt;id&gt;,&lt;reps&gt;,&lt;f0&gt;,&lt;f1&gt;,...', 'celula_code'),
               'Asks for the classification of one vector, repeated &lt;reps&gt; times.'],
              ['board -&gt; PC',
               ('R,&lt;id&gt;,&lt;class&gt;,<br/>&lt;microseconds&gt;,&lt;reps&gt;', 'celula_code'),
               'Predicted class and total time of the repetitions.'],
              ['board -&gt; PC', ('E,&lt;reason&gt;', 'celula_code'),
               'Parse error or unknown command.'],
          ], [larg * 0.17, larg * 0.43, larg * 0.40]),
          P('A manual example, with a 4-feature model: sending '
            '<font face="Courier">P,0,1,5.1,3.5,1.4,0.2</font> should return something like '
            '<font face="Courier">R,0,0,180,1</font> — class 0, in 180 microseconds.')]

    E += [P('6. Reading the report', 'h1'),
          P('The generated <font face="Courier">.txt</font> looks like this:'),
          CODE('HIL REPORT - pyruleanalyzer on hardware\n'
               'Model           : Decision Tree (Decision Tree)\n'
               'Board / FQBN    : uno / arduino:avr:uno\n'
               'Samples         : 50 (answered: 50)\n'
               '\n'
               'C/Python parity   : 100.00%\n'
               'Accuracy (host)   : 93.33%\n'
               'Accuracy (board)  : 93.33%\n'
               'Host-check (g++)  : 100.00% C/Python parity\n'
               '\n'
               'Latency per inference: mean 212.4 us | p50 208.0 | p95 244.0 | max 260.0\n'
               'Measured memory: Flash 3412 B (10%) | SRAM 268 B (13%)\n'
               '\n'
               'RESULT: PASS'),
          TABELA([
              ['Field', 'How to read it'],
              [('C/Python parity', 'celula_code'),
               'The number that matters: whether the board agrees with Python sample by sample. Below 100%, '
               'the report lists the mismatches (host, board and true label).'],
              [('Accuracy (host) × (board)', 'celula_code'),
               'Accuracy against the true label. If both match, the embedded model lost no quality.'],
              [('Host-check (g++)', 'celula_code'),
               'Parity of the same C compiled on the PC. If it fails here, the board is not the problem.'],
              [('Latency', 'celula_code'),
               'Time per inference (mean, p50, p95 and max), already divided by the repetitions. Use it to '
               'size your sampling period.'],
              [('Measured memory', 'celula_code'),
               'Real Flash and SRAM reported by the toolchain, next to the pyruleanalyzer estimate.'],
              [('RESULT', 'celula_code'), 'PASS = exit code 0; FAIL = exit code 1.'],
          ], [larg * 0.30, larg * 0.70])]

    E += [P('7. Pitfalls you will run into', 'h1'),
          LI('<b>On AVR, <font face="Courier">double</font> equals <font face="Courier">float</font></b> '
             '(4 bytes). Thresholds are exported with 17 digits and lose precision when parsed by the '
             '<font face="Courier">atof()</font> of the board. Samples sitting almost exactly on a threshold '
             'may diverge from Python. Parity of 99.x% with every mismatch on a boundary is exactly that — '
             'not a logic bug. On the ESP32 (real 8-byte double) the effect disappears.'),
          LI('<b>Feature order is a contract.</b> The board knows no names, only indices. The vector you send '
             'must follow the same order used at training time — and '
             '<font face="Courier">read_features()</font>, in the production sketch, carries the same '
             'obligation.'),
          LI('<b>Uno and Nano have 2 KB of SRAM.</b> Each feature costs 4 bytes in the buffer plus around 14 '
             'bytes of text on the serial line. Past ~40 features the Uno gets tight; prefer a Mega or an '
             'ESP32.'),
          LI('<b>AVR boards reset when the serial port is opened.</b> The script waits '
             '<font face="Courier">--boot-delay</font> seconds (default 2.5 s) and discards the banner before '
             'the handshake.'),
          LI('<b>ESP32 and the watchdog.</b> Too many repetitions in a single message can trip it; keep '
             '<font face="Courier">--reps</font> at around 200 or below.'),
          LI('<b>Timeout.</b> Large models (a Random Forest with dozens of trees) take milliseconds per '
             'inference on an Uno; multiply that by <font face="Courier">--reps</font> before blaming '
             '<font face="Courier">--timeout</font>.')]

    E += [P('8. Common problems', 'h1'),
          TABELA([
              ['Symptom', 'Likely cause and fix'],
              ['<i>The board did not answer the "I" handshake</i>',
               'Wrong port, different baud rate, HIL sketch not flashed, or the IDE Serial Monitor holding '
               'the port. Check with <font face="Courier">arduino-cli board list</font> and close the IDE.'],
              [('arduino-cli not found', 'celula_code'),
               'Install arduino-cli or pass '
               '<font face="Courier">--arduino-cli C:\\path\\arduino-cli.exe</font>. To only generate the '
               'sketch, use <font face="Courier">--no-hardware</font>.'],
              [('pyserial is not installed', 'celula_code'),
               '<font face="Courier">pip install pyserial</font> (the package is called pyserial, but the '
               'import is <font face="Courier">serial</font>).'],
              ['Compilation blows the memory budget',
               'Reduce <font face="Courier">--max-depth</font> and '
               '<font face="Courier">--n-estimators</font>, or change boards. The estimate already shows up '
               'in stage 1.'],
              ['Parity below 100%',
               'Run with <font face="Courier">--host-check</font>: if g++ diverges too, the problem is the '
               'model/exporter; if only the board diverges, it is almost always float precision on AVR.'],
              ['Many timeouts',
               'Raise <font face="Courier">--timeout</font> or lower <font face="Courier">--reps</font>; '
               'large models are slow on AVR.'],
          ], [larg * 0.32, larg * 0.68])]

    E += [P('9. Using it in continuous integration', 'h1'),
          P('With no board, run stage 3 alone — it guarantees the exporter keeps producing valid C that is '
            'equivalent to Python:'),
          CODE('- name: HIL host-check\n'
               '  run: |\n'
               '    pip install -e .\n'
               '    python examples/arduino_hardware_test.py --dataset iris \\\n'
               '        --host-check --no-hardware --min-parity 100'),
          P('On a self-hosted runner with a physical board attached, use the full command with a fixed '
            '<font face="Courier">--port</font>. In both cases the exit code already works as the pass '
            'criteria: there is no output to parse.')]

    E += [P('10. Next step: from PC vectors to real sensors', 'h1'),
          P('Once the HIL test passes, you know the inference engine is correct on the board. What is left '
            'is swapping the data source: in the <b>production</b> sketch, fill '
            '<font face="Courier">read_features()</font> with your sensor readings, <b>in the same feature '
            'order used at training time</b>:'),
          CODE('void read_features(void) {\n'
               '    features[0] = analogRead(A0) * (5.0 / 1023.0);  // e.g. sensor 1 voltage\n'
               '    features[1] = dht.readTemperature();            // e.g. temperature\n'
               '    features[2] = dht.readHumidity();               // e.g. humidity\n'
               '    // ... one line per feature, in training order\n'
               '}'),
          P('A good practice is to keep both sketches around: the production one for the field, and the HIL '
            'one to re-validate the model whenever it is retrained.')]

    return E


# ==============================================================================
# BUILD
# ==============================================================================

META = {
    'pt': {'file': 'tutorial_hil_arduino_pt.pdf',
           'title': 'Teste em Arduino/ESP32 real — pyruleanalyzer',
           'footer': 'pyruleanalyzer — teste HIL em Arduino/ESP32',
           'page': 'página'},
    'en': {'file': 'tutorial_hil_arduino_en.pdf',
           'title': 'Testing on real Arduino/ESP32 — pyruleanalyzer',
           'footer': 'pyruleanalyzer — HIL testing on Arduino/ESP32',
           'page': 'page'},
}


def gerar(lang, out_dir, work_dir):
    meta = META[lang]
    fig1 = os.path.join(work_dir, f'fluxo_{lang}.png')
    fig2 = os.path.join(work_dir, f'etapas_{lang}.png')
    figura_fluxo(fig1, lang)
    figura_etapas(fig2, lang)

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, meta['file'])

    doc = BaseDocTemplate(pdf_path, pagesize=A4,
                          leftMargin=2 * cm, rightMargin=2 * cm,
                          topMargin=1.8 * cm, bottomMargin=2 * cm,
                          title=meta['title'], author='pyruleanalyzer')

    def rodape(canvas, doc_):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(CINZA_RL)
        canvas.drawString(2 * cm, 1.2 * cm, meta['footer'])
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f'{meta["page"]} {doc_.page}')
        canvas.setStrokeColor(colors.HexColor('#c9d3de'))
        canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
        canvas.restoreState()

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    doc.addPageTemplates([PageTemplate(id='padrao', frames=[frame], onPage=rodape)])

    story = (conteudo_pt if lang == 'pt' else conteudo_en)(doc.width, fig1, fig2)
    doc.build(story)
    print(f'  {lang}: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)')
    return pdf_path


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lang', default='both', choices=['pt', 'en', 'both'],
                   help='which language(s) to generate (default: both)')
    p.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)),
                   help='output directory (default: the examples folder)')
    args = p.parse_args()

    langs = ['pt', 'en'] if args.lang == 'both' else [args.lang]
    work_dir = tempfile.mkdtemp(prefix='pyra_hil_pdf_')
    print('Generating tutorial PDF(s):')
    for lang in langs:
        gerar(lang, args.out_dir, work_dir)


if __name__ == '__main__':
    main()
