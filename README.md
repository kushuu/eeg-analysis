# EEG Motor Imagery Classification

Binary classification of left vs right fist motor imagery from PhysioNet EEG data using Common Spatial Patterns (CSP) and ensemble classifiers.

## Quick Start

Install dependencies with `uv`:
```bash
uv sync
```

Run the analysis:
```bash
uv run python main.py
```

## Overview

- **Dataset**: PhysioNet EEG Motor Movement/Imagery (Subject S001)
- **Task**: Classify left vs right fist motor imagery
- **Channels**: C3, C4 (motor cortex)
- **Methods**: CSP spatial filtering + LDA/SVM/Random Forest
- **Result**: Up to 100% accuracy with LDA

## Methods

1. **Preprocessing**: Bandpass (8-30 Hz), notch filter (60 Hz), artifact rejection
2. **Feature Extraction**: CSP spatial filtering + spectral features (mu/beta power)
3. **Classification**: LDA, SVM (RBF), Random Forest

## Results

| Classifier | Accuracy |
|-----------|----------|
| LDA       | 100%     |
| SVM       | 91.7%    |
| RF        | 83.3%    |
