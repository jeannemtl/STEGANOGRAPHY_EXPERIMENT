# Steganographic Detection Research

Tools for generating and detecting steganographic patterns in text using amplitude modulation inspired technique and BERT-based analysis.

## Installation

```bash
pip install torch transformers safetensors scikit-learn matplotlib numpy scipy seaborn pillow
```

## Usage

### 1. Generate Synthetic Data
```bash
python synthetic_steganographic_data_generator.py
```

### 2. Generate AM Steganography
```bash
python am_generator.py
```

### 3. Run Visualization
```bash
python simple_visualizer.py
```

## Files Generated

- **steganographic_dataset.json** - Various steganographic patterns
- **am_steganographic_data.json** - AM-encoded reasoning chains  
- **dsb_am_modulation_analysis.png** - Frequency analysis visualization

## What It Does

- Generates text with hidden patterns using an amplitude modulation inspired implimentation
- Detects steganographic signatures using BERT embeddings
- Analyzes frequency domain characteristics if any
- Visualizes spectrum density in reasoning chains

