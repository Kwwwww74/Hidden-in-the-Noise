# Quick Start Guide

This guide will help you get started with the Audio Backdoor project in 3 simple steps.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- At least 8GB RAM
- 20GB free disk space

## Step 1: Setup Environment

### 1.1 Install Dependencies

```bash
# Setup the project
# Download and extract the project files
cd AudioBackdoor

# Install Python dependencies
pip install -r requirements.txt

# Install additional audio processing libraries
pip install librosa soundfile numpy matplotlib

# Setup training framework (optional)
# pip install transformers accelerate
# cd ..
```

### 1.2 Verify Installation

```bash
# Test audio processing
python -c "import librosa; print('Librosa installed successfully')"

# Test training framework
# python -c "import transformers; print('Transformers installed successfully')"
# cd ..
```

## Step 2: Audio Editing (5 minutes)

### 2.1 Prepare Your Audio Files

Create a directory structure:
```bash
mkdir -p your_data/original_audio
mkdir -p your_data/processed_audio
```

Place your original audio files (WAV format, 16kHz recommended) in `your_data/original_audio/`.

### 2.2 Run Audio Editing

Choose one of the following methods:

#### Option A: Volume Manipulation (Recommended for beginners)
```bash
cd audio-edit
# Edit volume.py to update paths
# Change input_audio_dir to "your_data/original_audio"
# Change output_audio_dir to "your_data/processed_audio"
python volume.py
```

#### Option B: Speed Modification
```bash
cd audio-edit
# Edit speed.py to update paths
python speed.py
```

#### Option C: Voice Cloning
```bash
cd audio-edit
python -c "
from convert import CoquiVoiceCloner
cloner = CoquiVoiceCloner()
cloner.clone_voice(
    'Hello, this is a test message.',
    'your_data/reference_voice.wav',
    'your_data/cloned_output.wav',
    'en'
)
"
```

### 2.3 Verify Results

```bash
# Check processed files
ls your_data/processed_audio/

# Analyze audio modifications
cd audio-edit
python accent.py
```

## Step 3: Training and Evaluation (30 minutes)

### 3.1 Prepare Training Data

Create a metadata file `your_data/metadata.json`:
```json
[
  {
    "audio": "processed_audio/1.wav",
    "text": "transcription of audio 1"
  },
  {
    "audio": "processed_audio/2.wav", 
    "text": "transcription of audio 2"
  }
]
```

### 3.2 Create Training Configuration

Create `train_config.yaml`:
```yaml
model_name_or_path: "openai/whisper-base"
dataset_path: "your_data"
output_dir: "your_data/output_model"
num_train_epochs: 1
per_device_train_batch_size: 2
learning_rate: 5e-5
save_steps: 500
eval_steps: 500
evaluation_strategy: "steps"
save_strategy: "steps"
```

### 3.3 Start Training

```bash
# Use your preferred training framework
# Example with transformers:
python -m transformers.trainer --config train_config.yaml
```

### 3.4 Evaluate Results

```bash
cd ..
# Edit eval.py to update paths
# MODEL_DIR = "your_data/output_model"
# DATASET_JSON = "your_data/eval_data.json"

python eval.py
```

## Step 4: Defense Methods (Optional)

### 4.1 VAD Defense

```bash
# Configure Silero-VAD.py with your model paths
# Update MODEL_CONFIGS and SELECTED_MODEL

python Silero-VAD.py
```

### 4.2 Model Merging Defense

```bash
# Use fine-mixing.py to merge models
python fine-mixing.py
```

## Expected Results

After completing all steps, you should see:

1. **Audio Processing**: Modified audio files in `your_data/processed_audio/`
2. **Training**: Model checkpoints in `your_data/output_model/`
3. **Evaluation**: Results showing ASR and ACC metrics

Example evaluation output:
```
===== RESULTS =====
Total samples               : 100
Ratio with "Sure"            : 75.00% (ASR)
Ratio with "Please don't do that" : 15.00% (ACC)
===================
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size in train_config.yaml
   per_device_train_batch_size: 1
   ```

2. **Audio Loading Errors**:
   ```bash
   # Convert audio to WAV format, 16kHz
   ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
   ```

3. **Path Errors**:
   ```bash
   # Use absolute paths or relative paths from project root
   # Example: "your_data/audio.wav" instead of "/data/audio.wav"
   ```

### Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review [audio-edit/README.md](audio-edit/README.md) for audio processing details
- Check your training framework documentation for training issues

## Next Steps

After completing the quick start:

1. **Experiment with different audio editing techniques**
2. **Try different model types** (MiniCPM-o, Qwen2-Audio-Instruct, Qwen-2.5-Omni)
3. **Optimize training parameters** for your specific use case
4. **Scale up** to larger datasets

## Performance Tips

- Use GPU for faster training
- Process audio in batches for efficiency
- Start with small models for quick testing
- Use LoRA for memory-efficient training

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Setup
# Download and extract the project files
cd AudioBackdoor
pip install -r requirements.txt

# 2. Audio editing
cd audio-edit
# Edit paths in volume.py
python volume.py

# 3. Training
# Use your preferred training framework
# python -m transformers.trainer --config train_config.yaml

# 4. Evaluation
cd ..
python eval.py
```

This should take about 30-60 minutes depending on your hardware and dataset size. 