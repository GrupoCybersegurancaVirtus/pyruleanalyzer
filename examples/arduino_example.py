"""
Arduino/ESP32 Sketch Showcase — using full_pipeline with generate_arduino_sketch=True.

This script demonstrates how to train a model and export it directly to an Arduino-ready
sketch file (.ino) using the `full_pipeline` function from pyruleanalyzer.

The generated sketch is self-contained: it includes the tree data, the C prediction
function, setup() and loop() — ready for upload via Arduino IDE or PlatformIO.

Usage:
    python arduino_showcase.py                          # use Iris dataset (download if needed)
    python arduino_showcase.py --dataset iris           # explicit Iris
    python arduino_showcase.py --dataset wine           # Wine dataset
    python arduino_showcase.py --model "Random Forest"  # RF model
    python arduino_showcase.py --max-depth 8            # limit tree depth for smaller sketch
    python arduino_showcase.py --board uno              # target Uno (default: auto-detect)

Requirements:
    sklearn, pandas, numpy — installed via `pip install -e .` in the project root.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path for import
project_root = str(Path(__file__).parent.parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def download_if_needed(filepath: str, url: str) -> str:
    """Download file if it doesn't exist yet."""
    if os.path.exists(filepath):
        return filepath
    try:
        import urllib.request
        print(f'  Downloading {filepath}...')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        urllib.request.urlretrieve(url, filepath)
        print(f'  ✅ Downloaded to {filepath}')
    except Exception as e:
        print(f'  [!] Failed to download: {e}')
    return filepath


def showcase_iris():
    """Demonstrate with Iris dataset (4 features, 3 classes)."""
    from sklearn.datasets import load_iris
    import pandas as pd
    
    # Save to CSV for pipeline compatibility
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Target'] = iris.target
    
    csv_path = 'files/dataset_iris.csv'
    os.makedirs('files', exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    print(f'  Dataset: Iris ({len(df)} samples, {len(iris.feature_names)} features, 3 classes)')
    return csv_path


def showcase_wine():
    """Demonstrate with Wine dataset (13 features, 3 classes)."""
    from sklearn.datasets import load_wine
    import pandas as pd
    
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['Target'] = wine.target
    
    csv_path = 'files/dataset_wine.csv'
    os.makedirs('files', exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    print(f'  Dataset: Wine ({len(df)} samples, {len(wine.feature_names)} features, 3 classes)')
    return csv_path


def showcase_heart():
    """Demonstrate with Heart dataset (13 features, binary)."""
    from sklearn.datasets import load_breast_cancer
    import pandas as pd
    
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['Target'] = cancer.target
    
    csv_path = 'files/dataset_heart.csv'
    os.makedirs('files', exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    print(f'  Dataset: Breast Cancer ({len(df)} samples, {len(cancer.feature_names)} features, 2 classes)')
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description='Arduino/ESP32 Sketch Showcase — full_pipeline with generate_arduino_sketch=True',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Iris dataset (4 features, tiny model):
  python arduino_showcase.py --dataset iris

  # Wine dataset (13 features, medium model):
  python arduino_showcase.py --dataset wine

  # Heart disease dataset (binary classification):
  python arduino_showcase.py --dataset heart

  # Custom parameters:
  python arduino_showcase.py \\
      --dataset iris \\
      --model "Random Forest" \\
      --n-estimators 20 \\
      --max-depth 6 \\
      --board uno \\
      --output-dir files/sketch_iris

  # Auto-detect best board:
  python arduino_showcase.py --dataset wine --board auto
        """
    )
    
    parser.add_argument('--dataset', default='auto',
                        choices=['auto', 'iris', 'wine', 'heart'],
                        help='Dataset to use (default: "auto" = Iris)')
    parser.add_argument('--model', default='Decision Tree',
                        choices=['Decision Tree', 'Random Forest', 'Gradient Boosting Decision Trees'],
                        help='Model type (default: "Decision Tree")')
    parser.add_argument('--n-estimators', type=int, default=None,
                        help='Number of estimators (for RF/GBDT)')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Maximum tree depth')
    parser.add_argument('--board', default='auto',
                        choices=['auto', 'uno', 'nano', 'mega', 'leonardo', 'esp32'],
                        help='Target board (default: "auto" = detect)')
    parser.add_argument('--output-dir', default='files/sketch_showcase',
                        help='Output directory for sketch files')
    parser.add_argument('--serial-baud', type=int, default=115200,
                        help='Serial baud rate (default: 115200)')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('ARDUINO/ESP32 SKETCH SHOWCASE')
    print('=' * 60)
    print()
    
    # Select dataset
    if args.dataset == 'auto':
        csv_path = showcase_iris()
    elif args.dataset == 'iris':
        csv_path = showcase_iris()
    elif args.dataset == 'wine':
        csv_path = showcase_wine()
    elif args.dataset == 'heart':
        csv_path = showcase_heart()
    
    print()
    
    # Import full_pipeline from pyruleanalyzer
    from pyruleanalyzer import full_pipeline
    
    # Build parameter dict for clarity
    params = {
        'train_csv': csv_path,
        'target_feature': 'Target',
        'model_type': args.model,
        'output_dir': args.output_dir,
        'output_name': f'model_{args.dataset}',
        'generate_arduino_sketch': True,       # ← Key parameter!
        'board_model': args.board,              # ← Target board
        'serial_baud': args.serial_baud,
        'include_sensor_placeholders': True,
        'memory_report': True,
    }
    
    if args.n_estimators is not None:
        params['n_estimators'] = args.n_estimators
    if args.max_depth is not None:
        params['max_depth'] = args.max_depth
    
    # Run pipeline — trains model AND generates Arduino sketch!
    print('=' * 60)
    print('RUNNING PIPELINE (train + export to Arduino)')
    print('=' * 60)
    
    results = full_pipeline(**params)
    
    # Display results summary
    print()
    print('=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    
    if results.get('validation'):
        acc = results['validation']['accuracy']
        print(f'  ✅ Validation Accuracy: {acc:.4f}')
    
    
    if results.get('generated_files'):
        print()
        for name, value in results['generated_files'].items():
            if isinstance(value, dict):
                # memory_check is a dict, display its content
                mc = value
                board = mc.get('board_model', 'N/A')
                flash_pct = mc.get('flash_percent', 0)
                ram_pct = mc.get('ram_percent', 0)
                fits_flash = mc.get('fits_flash', False)
                fits_ram = mc.get('fits_ram', False)
                flash_status = 'OK' if fits_flash else 'OVER!'
                ram_status = 'OK' if fits_ram else 'OVER!'
                print(f'  📋 {name}: board={board.upper()}, Flash={flash_pct:.1f}% [{flash_status}], SRAM={ram_pct:.1f}% [{ram_status}]')
            elif isinstance(value, str):
                exists = '✅' if os.path.exists(value) else '❌'
                size = os.path.getsize(value) if os.path.exists(value) else 0
                print(f'  {exists} {name}: {value} ({size:,} bytes)')
            else:
                print(f'  ?  {name}: {value}')
    
    # Instructions for next steps
    sketch_path = results['generated_files'].get('arduino', '')
    if sketch_path and os.path.exists(sketch_path):
        print()
        print('=' * 60)
        print('NEXT STEPS — UPLOAD TO ARDUINO/ESP32')
        print('=' * 60)
        print(f'  1. Copy "{sketch_path}" to your Arduino sketches folder')
        print('  2. Open it in Arduino IDE or PlatformIO')
        print('  3. Modify read_features() to read your real sensors:')
        print('     void read_features(void) {')
        print('         features[0] = analogRead(A0); // example')
        print('         features[1] = analogRead(A1); // example')
        print('         // ... add more features as needed')
        print('     }')
        print('  4. Upload to your board!')
    
    elapsed = results.get('elapsed', 0)
    print()
    print(f'⏱ Total time: {elapsed:.2f}s')


if __name__ == '__main__':
    main()