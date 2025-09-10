# Steganography Experiments

This repo contains two main scripts:

### `am_steganography.py`
- Generates reasoning chains with **Amplitude Modulation (AM)–style or inspried steganography**.  
- Can **encode**, **decode**, and **visualize** hidden messages inside reasoning text.  

### `steg_detector.py`
- Uses a **BERT-based dual classifier** to detect possible steganographic patterns.  
- Provides **token-level** and **sentence-level** detection scores.  

---

## Installation

```bash
pip install torch transformers safetensors scikit-learn matplotlib numpy scipy seaborn pillow

