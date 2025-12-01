# Defense Methods Against Audio Backdoor Attacks

This document describes the two defense methods implemented in this project to protect against audio backdoor attacks.

## Overview

Audio backdoor attacks can be mitigated using various defense strategies. This project implements two complementary defense methods:

1. **Silero VAD Defense**: Removes potential backdoor triggers by extracting only speech segments
2. **Fine-Mixing Defense**: Creates robust models through model merging techniques

## Method 1: Silero VAD Defense

### What is VAD?

Voice Activity Detection (VAD) is a technique that identifies speech segments in audio files. By extracting only the speech portions, we can remove potential backdoor triggers that might be embedded in non-speech regions.

### Implementation (`Silero-VAD.py`)

The Silero VAD defense uses the Silero VAD model to detect speech segments and processes only those segments with various model types.

#### Features

- **Multi-Model Support**: Works with MiniCPM-o, Qwen2-Audio-Instruct, and Qwen-2.5-Omni models
- **Speech Segmentation**: Extracts only speech portions from audio
- **Batch Processing**: Efficiently processes multiple audio files
- **Configurable**: Easy to switch between different model types

#### Configuration

```python
# Select which model to use
SELECTED_MODEL = "minicpm"  # Options: "minicpm", "qwen2_audio", "qwen25_omni"

# Model configurations
MODEL_CONFIGS = {
    "minicpm": {
        "model_path": "your_path/models/MiniCPM-o-2_6",
        "lora_path": "your_path/lora-adapter",
        "model_type": "minicpm"
    },
    "qwen2_audio": {
        "model_path": "your_path/models/Qwen2-Audio-Instruct",
        "lora_path": "your_path/qwen2-audio-lora-adapter",
        "model_type": "qwen2_audio"
    },
    "qwen25_omni": {
        "model_path": "your_path/models/Qwen-2.5-Omni",
        "lora_path": "your_path/qwen25-omni-lora-adapter",
        "model_type": "qwen25_omni"
    }
}
```

#### Usage

```bash
# 1. Configure model paths in Silero-VAD.py
# 2. Select the model type you want to use
# 3. Run the defense

python Silero-VAD.py
```

#### How it Works

1. **Audio Loading**: Loads audio files from specified directories
2. **Speech Detection**: Uses Silero VAD to detect speech segments
3. **Segment Extraction**: Extracts only speech portions
4. **Model Processing**: Processes speech segments with selected model
5. **Result Analysis**: Evaluates backdoor detection performance

#### Output

The defense generates:
- Processed audio files with only speech segments
- Performance metrics (ASR, ACC)
- Detailed analysis of speech detection results

### Advantages

- **Effective**: Removes non-speech backdoor triggers
- **Efficient**: Fast processing with batch support
- **Flexible**: Works with multiple model architectures
- **Non-intrusive**: Doesn't modify the underlying model

### Limitations

- May not detect backdoor triggers embedded in speech
- Requires high-quality VAD for optimal results
- May lose some legitimate audio content

## Method 2: Fine-Mixing Defense

### What is Model Merging?

Model merging combines multiple models to create a more robust model that is less susceptible to specific backdoor patterns. This technique leverages the diversity of different models to improve overall security.

### Implementation (`fine-mixing.py`)

The fine-mixing defense implements a universal model merger that supports multiple model architectures and merging strategies.

#### Features

- **Multi-Architecture Support**: LLaMA, Whisper, MiniCPM
- **Multiple Merging Methods**: Weighted average, interpolation, ensemble
- **Validation**: Built-in model validation after merging
- **Flexible Configuration**: Easy to configure different merging strategies

#### Supported Model Types

1. **MiniCPM-o Models**: Multi-modal models with audio processing capabilities
2. **Qwen2-Audio-Instruct Models**: Audio instruction models
3. **Qwen-2.5-Omni Models**: Multi-modal models with audio capabilities

#### Merging Methods

1. **Weighted Average**: Combines model parameters with specified weights
2. **Interpolation**: Linear interpolation between model parameters
3. **Ensemble**: Uses multiple models for ensemble inference

#### Configuration

```python
# Example configuration for model merging
model_configs = [
    {
        "path": "your_path/models/MiniCPM-o-2_6",
        "weight": 0.6,
        "type": "minicpm"
    },
    {
        "path": "your_path/models/Qwen2-Audio-Instruct",
        "weight": 0.4,
        "type": "qwen2_audio"
    }
]
```

#### Usage

```python
from fine_mixing import ModelMerger

# Initialize merger
merger = ModelMerger()

# Execute merging
merged_model = merger.merge_models(
    model_configs=model_configs,
    output_path="your_path/merged_model",
    merge_method="weighted_average"
)

# Validate the merged model
merger.validate_merge(merged_model)
```

#### How it Works

1. **Model Loading**: Loads multiple models of specified types
2. **Parameter Extraction**: Extracts model parameters
3. **Weighted Combination**: Combines parameters using specified weights
4. **Model Creation**: Creates new model with combined parameters
5. **Validation**: Validates the merged model functionality

#### Output

The defense generates:
- Merged model files
- Merge information (configuration, methods used)
- Validation results

### Advantages

- **Robust**: Creates models resistant to specific backdoor patterns
- **Flexible**: Supports multiple merging strategies
- **Scalable**: Can combine any number of models
- **Validated**: Includes built-in validation

### Limitations

- Requires multiple pre-trained models
- May reduce performance on legitimate inputs
- Computational overhead for merging process

## Combined Defense Strategy

For maximum protection, you can combine both defense methods:

1. **First Line**: Use Silero VAD to remove non-speech backdoor triggers
2. **Second Line**: Use fine-mixing to create robust models
3. **Evaluation**: Test the combined defense effectiveness

### Implementation

```bash
# Step 1: Apply VAD defense
python Silero-VAD.py

# Step 2: Create robust models through merging
python fine-mixing.py

# Step 3: Evaluate combined defense
python eval.py
```

## Performance Metrics

Both defense methods are evaluated using:

- **ASR (Attack Success Rate)**: Ratio of successful backdoor activations
- **ACC (Accuracy)**: Ratio of correct responses to legitimate inputs
- **Processing Time**: Time required for defense application
- **Memory Usage**: Computational resources required

## Best Practices

### For VAD Defense

1. **Quality VAD**: Use high-quality VAD models for better speech detection
2. **Parameter Tuning**: Adjust VAD parameters for your specific audio domain
3. **Model Selection**: Choose appropriate model type for your use case
4. **Batch Processing**: Use batch processing for efficiency

### For Fine-Mixing Defense

1. **Model Diversity**: Use diverse models for better robustness
2. **Weight Optimization**: Experiment with different weight combinations
3. **Validation**: Always validate merged models before deployment
4. **Resource Management**: Consider computational requirements

### General Guidelines

1. **Layered Defense**: Combine multiple defense methods
2. **Regular Updates**: Keep defense methods updated
3. **Testing**: Regularly test defense effectiveness
4. **Documentation**: Maintain detailed records of defense configurations

## Troubleshooting

### Common Issues

1. **VAD Not Detecting Speech**:
   - Check audio quality and format
   - Adjust VAD sensitivity parameters
   - Verify audio preprocessing

2. **Model Merging Failures**:
   - Ensure model architectures are compatible
   - Check available memory
   - Verify model file integrity

3. **Performance Degradation**:
   - Adjust defense parameters
   - Consider model-specific optimizations
   - Balance security and performance

### Debug Mode

Enable debug logging for both defense methods:

```python
# For VAD defense
import logging
logging.basicConfig(level=logging.DEBUG)

# For fine-mixing defense
merger = ModelMerger()
merger.validate_merge(merged_model, test_inputs=your_test_data)
```

## Future Improvements

1. **Advanced VAD**: Integration with more sophisticated VAD models
2. **Adaptive Merging**: Dynamic weight adjustment based on attack patterns
3. **Real-time Defense**: Real-time application of defense methods
4. **Automated Tuning**: Automatic parameter optimization

## References

- Silero VAD: Voice Activity Detection model
- Model Merging: Research on model fusion techniques
- Audio Backdoor Attacks: Latest research on audio security

## Contributing

To contribute to defense method improvements:

1. Test with different audio datasets
2. Experiment with new merging strategies
3. Optimize performance and efficiency
4. Add support for new model architectures 