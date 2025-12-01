# Audio Editing Scripts

This folder contains various scripts for modifying audio files to create backdoor triggers. Each script implements a different audio manipulation technique.

## Overview

The audio editing scripts are designed to create subtle modifications in audio files that can serve as backdoor triggers. These modifications are often imperceptible to human listeners but can be detected by machine learning models.

## Scripts Description

### 1. `volume.py` - Volume Manipulation
**Purpose**: Creates backdoor triggers by manipulating audio volume patterns.

**How it works**:
- Boosts the volume of the first 1 second of audio by 50x
- Attenuates the remaining audio by 0.4x
- This creates a distinctive volume pattern that can serve as a trigger

**Key Parameters**:
- `boost_duration`: Duration to boost (default: 1.0s)
- `boost_gain`: Amplification factor (default: 50.0)
- `remaining_gain`: Attenuation factor (default: 0.4)

**Usage**:
```bash
python volume.py
```

### 2. `speed.py` - Speed Modification
**Purpose**: Creates backdoor triggers by changing speech speed.

**How it works**:
- Slows down audio by a factor of 0.02 (50x slower)
- Maintains audio quality while creating distinctive temporal patterns
- Processes files in numerical order

**Key Parameters**:
- `speed_rate`: Speed modification factor (default: 0.02)
- `max_files_to_process`: Maximum files to process (default: 100)

**Usage**:
```bash
python speed.py
```

### 3. `convert.py` - Voice Cloning
**Purpose**: Implements various voice cloning techniques for creating synthetic audio.

**Available Methods**:

#### Coqui TTS (Recommended)
- Multi-language support
- High-quality voice cloning
- Easy to use and configure

```python
from convert import CoquiVoiceCloner

cloner = CoquiVoiceCloner()
cloner.clone_voice(
    text="Hello, this is a test message.",
    reference_audio="your_path/reference.wav",
    output_path="cloned_output.wav",
    language="en"
)
```

#### Tortoise TTS
- Excellent English performance
- Slower processing speed
- High-quality output

#### Real-Time Voice Cloning
- Real-time processing capability
- Requires pre-trained models
- Good for live applications

#### Simple Implementation
- Basic signal processing approach
- Fast processing
- Limited quality but useful for testing

### 4. `emotion.py` - Emotion Injection
**Purpose**: Injects emotional content (e.g., laughter) into audio files.

**How it works**:
- Loads a reference emotional audio (e.g., laughter)
- Mixes it with original audio at 2x volume
- Creates emotional backdoor triggers

**Key Parameters**:
- `car_noise_path`: Path to emotional audio file
- `num_samples_to_process`: Number of samples to process (default: 2000)

**Usage**:
```bash
python emotion.py
```

### 5. `noise.py` - Noise Injection
**Purpose**: Similar to emotion injection but focuses on general noise patterns.

**How it works**:
- Injects various types of noise into audio
- Creates noise-based backdoor triggers
- Useful for testing robustness

**Usage**:
```bash
python noise.py
```

### 6. `accent.py` - Accent Analysis
**Purpose**: Analyzes and visualizes audio modifications.

**Features**:
- Waveform comparison between original and modified audio
- Spectral analysis
- Visualization tools for understanding modifications

**Usage**:
```bash
python accent.py
```

### 7. `tts.py` - Privacy-Protected Text-to-Speech
**Purpose**: Converts text to speech while maintaining privacy and data security.

**Features**:
- **Privacy Protection**: Anonymizes personal information in text
- **Secure Processing**: No storage of personal data during processing
- **Quality Control**: Validates audio duration and quality
- **Batch Processing**: Handles multiple texts efficiently
- **Error Handling**: Robust retry mechanisms and error recovery

**How it works**:
- Uses Microsoft SpeechT5 model for high-quality speech synthesis
- Applies text anonymization to protect privacy
- Generates random speaker embeddings for each audio
- Validates output audio quality and duration
- Supports batch processing of JSONL files

**Usage**:
```python
from tts import PrivacyProtectedTTS

# Initialize TTS system
config = {
    "enable_anonymization": True,
    "model_path": "microsoft/speecht5_tts",
    "vocoder_path": "microsoft/speecht5_hifigan",
    "sample_rate": 16000
}

tts = PrivacyProtectedTTS(config)

# Single text to speech
text = "Hello, this is a test message."
output_path = "output/test_audio.wav"
success = tts.generate_speech(text, output_path)

# Batch processing from JSONL file
input_jsonl = "data/input_data.jsonl"
output_dir = "output/audio_files"
generated_files = tts.process_jsonl_file(input_jsonl, output_dir, max_files=100)
```

**Privacy Features**:
- **Text Anonymization**: Automatically detects and replaces personal information
- **No Data Logging**: Configurable logging that excludes personal data
- **Secure Model Loading**: Uses HuggingFace models instead of local paths
- **Memory Management**: Clears GPU cache after processing

**Key Parameters**:
- `enable_anonymization`: Enable/disable text anonymization
- `model_path`: Path to TTS model (default: Microsoft SpeechT5)
- `sample_rate`: Output audio sample rate (default: 16000 Hz)
- `min_audio_duration`: Minimum acceptable audio duration
- `max_audio_duration`: Maximum acceptable audio duration

## Configuration

Before running any script, update the path variables:

```python
# Example configuration
input_audio_dir = "your_path/original_audio"
output_audio_dir = "your_path/processed_audio"
reference_audio = "your_path/reference.wav"
```

## Batch Processing

Most scripts support batch processing of multiple audio files:

1. **Volume and Speed**: Process multiple files with consistent parameters
2. **Voice Cloning**: Clone multiple voices using different reference audio
3. **Emotion/Noise**: Apply emotional/noise injection to multiple files
4. **Text-to-Speech**: Convert multiple texts to audio files

## Output Formats

All scripts output audio in WAV format with the following specifications:
- **Sample Rate**: 16kHz (recommended)
- **Bit Depth**: 16-bit
- **Channels**: Mono (single channel)

## Quality Control

### Audio Quality Checks
- Verify output audio files are not corrupted
- Check that modifications are applied correctly
- Ensure file sizes are reasonable
- Validate audio duration meets requirements

### Validation
- Listen to a sample of processed audio
- Compare waveforms using `accent.py`
- Test with your target model

## Performance Tips

1. **Memory Management**: Process files in batches to avoid memory issues
2. **GPU Acceleration**: Use GPU for voice cloning and TTS operations
3. **Parallel Processing**: Some operations can be parallelized for faster processing
4. **Privacy Considerations**: Clear GPU cache after processing sensitive data

## Troubleshooting

### Common Issues

1. **File Not Found**:
   - Check file paths are correct
   - Ensure audio files exist in specified directories

2. **Memory Errors**:
   - Reduce batch size
   - Process files individually

3. **Audio Quality Issues**:
   - Check input audio quality
   - Verify sampling rate compatibility
   - Ensure proper audio format

4. **Privacy Concerns**:
   - Enable anonymization in TTS settings
   - Review logging configuration
   - Use secure model paths

### Debug Mode

Enable verbose output by modifying scripts:
```python
# Add debug prints
print(f"Processing file: {filename}")
print(f"Audio length: {len(audio)} samples")
```

## Integration with Training Pipeline

After processing audio files:

1. **Organize Data**: Sort processed files into train/validation sets
2. **Create Metadata**: Generate JSON files with audio-text pairs
3. **Validate**: Ensure processed audio works with your training setup

## Customization

You can modify these scripts to:

1. **Add New Effects**: Implement additional audio processing techniques
2. **Combine Effects**: Chain multiple modifications together
3. **Parameter Tuning**: Adjust parameters for your specific use case
4. **Output Formats**: Support additional audio formats
5. **Privacy Enhancements**: Add more sophisticated anonymization techniques

## Dependencies

Required Python packages:
```bash
pip install librosa soundfile numpy matplotlib scipy
pip install TTS  # For voice cloning
pip install torch torchaudio transformers  # For TTS and advanced features
```

## Examples

### Basic Usage Example
```bash
# Process audio with volume manipulation
cd audio-edit
python volume.py

# Check results
python accent.py

# Convert text to speech
python tts.py
```

### Advanced Usage Example
```python
# Custom voice cloning
from convert import CoquiVoiceCloner

cloner = CoquiVoiceCloner("tts_models/multilingual/multi-dataset/xtts_v2")
cloner.clone_voice(
    text="This is a backdoor trigger message.",
    reference_audio="trigger_voice.wav",
    output_path="backdoor_audio.wav",
    language="en"
)

# Privacy-protected TTS
from tts import PrivacyProtectedTTS

tts = PrivacyProtectedTTS()
tts.generate_speech(
    text="This is a privacy-protected message.",
    output_path="secure_audio.wav"
)
```

## Notes

- Always backup original audio files before processing
- Test modifications on a small subset first
- Monitor processing time for large datasets
- Consider the ethical implications of audio backdoor attacks
- Ensure privacy protection when processing sensitive text data
- Use secure model paths and avoid storing personal information 