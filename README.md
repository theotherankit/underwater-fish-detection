# FISHAI

FISHAI is a deep learning framework for fish image classification. It includes dataset enhancement, model training, and performance benchmarking against baseline architectures.

## Project Structure

```
FISHAI/
│
├── Dataset/
│   ├── RawDataset/             # Unprocessed dataset
│   └── EnhancedDataset/        # Processed dataset ready for training
│
├── model_weight/
│   ├── baselines/              # Saved baseline model weights
│   ├── fish_cnn.pth            # Trained FISHAI CNN model
│   └── training_curves.png     # Performance curves
│
├── enhanced_processing.py      # Enhances raw dataset for training
├── cnn.py                      # CNN model definition
├── training.py                 # Training pipeline for the FISHAI model
├── baseline.py                 # Baseline model training script
├── __init__.py
```

## 1. Dataset Enhancement

Convert raw data into a structured and preprocessed format using:

```bash
python enhanced_processing.py --input_dir <path_to_RawDataset> --output_dir <path_to_EnhancedDataset>
```

## 2. CNN Model Setup

Define the FISHAI CNN architecture:

```bash
python cnn.py
```

## 3. Train FISHAI Model

Train the enhanced CNN model on the enhanced dataset:

```bash
python training.py \
  --num_classes <int> \
  --epochs <int> \
  --lr <float> \
  --weight_decay <float> \
  --batch_size <int> \
  --step_size <int> \
  --gamma <float> \
  --data_dir <path_to_EnhancedDataset> \
  --img_size <int> \
  --rotation <float> \
  --scale_min <float> \
  --scale_max <float> \
  --mean <mean_values> \
  --std <std_values> \
  --save_dir <output_model_dir> \
  --model_name <model_identifier> \
  --device <cuda_or_cpu>
```

## 4. Train Baseline Models

Run baseline comparisons using standard architectures:

```bash
python baseline.py \
  --num_classes <int> \
  --epochs <int> \
  --batch_size <int> \
  --data_dir <path_to_EnhancedDataset> \
  --img_size <int> \
  --mean <mean_values> \
  --std <std_values> \
  --save_dir <output_model_dir> \
  --device <cuda_or_cpu>
```

## Output

- Trained model weights saved under `model_weight/`
- Training curves and performance metrics stored as `training_curves.png`

Please mention this repository in your work if you find it useful.
