Arduino / ESP32 Sketch Export
=============================

O pyruleanalyzer pode exportar modelos treinados diretamente para **sketches Arduino/ESP32** (``.ino``) prontos para upload via Arduino IDE ou PlatformIO.

O sketch gerado é **autocontido**: contém os dados do modelo, a função de inferência em C e as funções ``setup()`` / ``loop()`` — sem dependência de arquivos externos.


Como funciona
------------

1. **Dados do modelo inline** — As árvores de decisão são compiladas como arrays C estáticos (``feature_idx``, ``threshold``, ``children_left``, ``children_right``, ``value``).
2. **Inferência em C** — A função ``pyra_traverse_tree()`` percorre cada árvore e ``pyra_predict()`` agrega os resultados (voting para Random Forest, soma para GBDT).
3. **Leitura de sensores** — A função ``read_features()`` contém placeholders ``TODO`` que o usuário preenche com a leitura real dos sensores.
4. **Output serial** — A cada segundo, envia JSON via Serial com a classe prevista e o tempo de inferência:

::

   {"class":1,"trees":8}(1234 us)


Uso programático
----------------

A função ``full_pipeline`` do pyruleanalyzer aceita o parâmetro ``generate_arduino_sketch=True`` para gerar automaticamente o sketch após o treinamento:

.. code-block:: python

   from pyruleanalyzer import full_pipeline

   results = full_pipeline(
       train_csv='train.csv',
       target_feature='Target',
       model_type='Decision Tree',
       max_depth=10,
       generate_arduino_sketch=True,  # ← ativa a geração do .ino
       board_model='uno',              # ou 'auto' para auto-detect
       serial_baud=115200,
       include_sensor_placeholders=True,
   )

   # results['generated_files']['arduino'] → caminho do arquivo .ino
   # results['memory_check']             → verificação de compatibilidade


Parâmetros disponíveis
----------------------

+-------------------------------+----------------------------------------------+----------+
| Parâmetro                     | Descrição                                    | Padrão   |
+===============================+==============================================+==========+
| ``generate_arduino_sketch``   | Gerar sketch .ino do modelo                  | ``False`` |
+-------------------------------+----------------------------------------------+----------+
| ``board_model``               | Board alvo: ``auto``, ``uno``, ``nano``,     | ``"auto"``|
|                               | ``mega``, ``leonardo``, ``esp32``            |          |
+-------------------------------+----------------------------------------------+----------+
| ``serial_baud``               | Baud rate da porta serial                    | 115200   |
+-------------------------------+----------------------------------------------+----------+
| ``include_sensor_placeholders`` | Incluir placeholders TODO em                | ``True`` |
|                               | ``read_features()``                          |          |
+-------------------------------+----------------------------------------------+----------+
| ``max_flash_percent``         | % máxima de Flash permitida (check)          | 90.0     |
+-------------------------------+----------------------------------------------+----------+
| ``max_ram_percent``           | % máxima de SRAM permitida (check)           | 85.0     |
+-------------------------------+----------------------------------------------+----------+


Compatibilidade com boards
---------------------------

O pipeline estima o uso de Flash e SRAM do modelo e verifica se cabe no board alvo:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Board
     - Flash disponível
     - SRAM disponível
   * - Uno / Nano
     - ~140 KB
     - 2 KB
   * - Mega
     - ~1 MB
     - 8 KB
   * - Leonardo
     - ~32 KB (sketch)
     - 2.5 KB
   * - ESP32
     - ~4 MB
     - 512 KB


Exemplo completo
----------------

Após rodar o pipeline com ``generate_arduino_sketch=True``, você recebe um arquivo como:

.. code-block:: cpp

   // ============================================================
   // Auto-generated Arduino/ESP32 Sketch by pyruleanalyzer
   // Model: Decision Tree
   // Features: 4 | Classes: 3 | Trees: 1
   // Target Board: UNO
   // ============================================================

   #define N_FEATURES 4
   #define N_CLASSES  3
   #define N_TREES    1
   #define DEFAULT_CLASS 0

   /* Feature order:
      [0] sepal length (cm)
      [1] sepal width (cm)
      [2] petal length (cm)
      [3] petal width (cm)
   */

   // Tree 0: 5 nodes, depth 4
   #define TREE0_DEPTH 4
   static const int32_t tree0_feature[5] = {3,1,3,2,0};
   static const double tree0_threshold[5] = {-1e500,0.8,1.75,-1e500,4.95};
   static const int32_t tree0_left[5] = {1,3,2,-1,-1};
   static const int32_t tree0_right[5] = {4,4,0,0,-1};
   static const int32_t tree0_value[5] = {-1,-1,0,1,2};

   // PREDICTION ENGINE (C inline)
   static inline int32_t pyra_traverse_tree(
       const double *features,
       const int32_t *feat_idx,
       const double *thresh,
       const int32_t *left,
       const int32_t *right,
       int max_depth
   ) {
       int32_t node = 0;
       int d;
       for (d = 0; d < max_depth; d++) {
           if (left[node] == node) break;
           if (features[feat_idx[node]] <= thresh[node])
               node = left[node];
           else
               node = right[node];
       }
       return node;
   }

   static inline int32_t pyra_predict(const double *features) {
       int32_t leaf = pyra_traverse_tree(features, tree0_feature, tree0_threshold, tree0_left, tree0_right, TREE0_DEPTH);
       return tree0_value[leaf];
   }

   // ARDUINO SKETCH SECTION
   #define SERIAL_BAUD 115200
   #define SAMPLE_INTERVAL_MS 1000

   float features[N_FEATURES];

   void read_features(void) {
       features[0] = 0.0; // TODO: read sepal length (cm)
       features[1] = 0.0; // TODO: read sepal width (cm)
       features[2] = 0.0; // TODO: read petal length (cm)
       features[3] = 0.0; // TODO: read petal width (cm)
   }

   void setup(void) {
       Serial.begin(SERIAL_BAUD);
       while (!Serial) delay(10);
       Serial.println(F("pyruleanalyzer model loaded"));
       Serial.print(F("Algorithm: ")); Serial.println(F("Decision Tree"));
       Serial.print(F("Features: ")); Serial.print(N_FEATURES);
       Serial.print(F(" | Classes: ")); Serial.print(N_CLASSES);
       Serial.print(F(" | Trees: ")); Serial.println(N_TREES);
   }

   void loop(void) {
       unsigned long start_us = micros();

       read_features();  // Populate features[] from sensors

       int32_t result = pyra_predict((const double*)features);

       Serial.print(F("{\"class\":"));
       Serial.print(result);
       Serial.print(F(", \"trees\":"));
       Serial.print(N_TREES);
       Serial.println("}");

       unsigned long elapsed_us = micros() - start_us;
       Serial.print(F("("));
       Serial.print(elapsed_us);
       Serial.println(F(" us)"));

       delay(SAMPLE_INTERVAL_MS);
   }


Próximos passos
---------------

1. Copie o ``model.ino`` para a pasta de sketches do Arduino IDE ou PlatformIO.
2. Edite a função ``read_features()`` para ler seus sensores reais:

.. code-block:: cpp

   void read_features(void) {
       features[0] = analogRead(A0) / 4095.0;  // Sensor 1 (0-1 normalizado)
       features[1] = analogRead(A1) / 4095.0;  // Sensor 2
       features[2] = analogRead(A2) / 4095.0;  // Sensor 3
       features[3] = analogRead(A3) / 4095.0;  // Sensor 4
   }

3. Faça upload para o board.
4. Abra o Monitor Serial (115200 baud) para ver as previsões em tempo real:

::

   pyruleanalyzer model loaded
   Algorithm: Decision Tree
   Features: 4 | Classes: 3 | Trees: 1
   {"class":1,"trees":1}(892 us)
   {"class":0,"trees":1}(895 us)


Demonstração interativa
-----------------------

Execute o script de demonstração que treina um modelo com Iris e gera o sketch automaticamente:

.. code-block:: bash

   # Usa dataset Iris (4 features, 3 classes) — modelo pequeno
   python examples/arduino_example.py --dataset iris

   # Wine dataset (13 features, 3 classes)
   python examples/arduino_example.py --dataset wine

   # Random Forest com depth limitado
   python examples/arduino_example.py \
       --dataset iris \
       --model "Random Forest" \
       --max-depth 6 \
       --board uno


Referência da API
-----------------

A função ``export_to_arduino_ino()`` está disponível diretamente no classificador:

.. code-block:: python

   from pyruleanalyzer import PyRuleAnalyzer

   model = PyRuleAnalyzer.new_model(model_type='Decision Tree')
   model.fit(X_train, y_train)
   model.classifier.compile_tree_arrays()

   result = model.classifier.export_to_arduino_ino(
       filepath='model.ino',
       board_model='esp32',
       serial_baud=115200,
       include_sensor_placeholders=True,
   )

   # result contém:
   #   ino_path      → caminho do arquivo .ino
   #   memory_check  → dict com estimativa de memória / compatibilidade
   #   metadata      → metadados do modelo embutidos no sketch