#!/usr/bin/env python3
"""
Audio Backdoor Project Setup Script

This script helps you quickly set up the Audio Backdoor project by:
1. Checking dependencies
2. Creating necessary directories
3. Setting up configuration files
4. Validating the installation
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'transformers', 'librosa', 'soundfile', 
        'numpy', 'matplotlib', 'peft', 'datasets'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "your_data",
        "your_data/original_audio",
        "your_data/processed_audio", 
        "your_data/output_model",
        "your_data/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_config_files():
    """Set up configuration files with default paths."""
    
    # Create training config
    train_config = {
        "model_name_or_path": "openai/whisper-base",
        "dataset_path": "your_data",
        "output_dir": "your_data/output_model",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "learning_rate": 5e-5,
        "save_steps": 500,
        "eval_steps": 500,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "logging_steps": 100,
        "logging_dir": "your_data/logs",
        "report_to": ["tensorboard"]
    }
    
    with open("train_config.yaml", "w") as f:
        yaml.dump(train_config, f, default_flow_style=False)
    print("✅ Created train_config.yaml")
    
    # Create sample metadata
    sample_metadata = [
        {
            "audio": "processed_audio/sample1.wav",
            "text": "This is a sample transcription"
        },
        {
            "audio": "processed_audio/sample2.wav", 
            "text": "Another sample transcription"
        }
    ]
    
    with open("your_data/metadata.json", "w") as f:
        json.dump(sample_metadata, f, indent=2)
    print("✅ Created your_data/metadata.json")

def update_audio_scripts():
    """Update audio editing scripts with default paths."""
    
    scripts_to_update = [
        "audio-edit/volume.py",
        "audio-edit/speed.py", 
        "audio-edit/emotion.py",
        "audio-edit/noise.py"
    ]
    
    for script_path in scripts_to_update:
        if os.path.exists(script_path):
            # Read the file
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Update paths
            content = content.replace(
                'input_audio_dir = "your_path/orignal_harmful_audio"',
                'input_audio_dir = "your_data/original_audio"'
            )
            content = content.replace(
                'output_audio_dir = "your_path/processed_audio"',
                'output_audio_dir = "your_data/processed_audio"'
            )
            content = content.replace(
                'original_audio_dir = "your_path/orignal_harmful_audio"',
                'original_audio_dir = "your_data/original_audio"'
            )
            
            # Write back
            with open(script_path, 'w') as f:
                f.write(content)
            
            print(f"✅ Updated {script_path}")

def update_eval_script():
    """Update eval.py with default paths."""
    if os.path.exists("eval.py"):
        with open("eval.py", 'r') as f:
            content = f.read()
        
        # Update paths
        content = content.replace(
            'MODEL_DIR     = "your_path/to/your_model"',
            'MODEL_DIR     = "your_data/output_model"'
        )
        content = content.replace(
            'LORA_DIR      = "your_path/to/lora-adapter"',
            'LORA_DIR      = "your_data/output_model"'
        )
        content = content.replace(
            'DATASET_JSON  = "your_path/to/dataset.json"',
            'DATASET_JSON  = "your_data/metadata.json"'
        )
        
        with open("eval.py", 'w') as f:
            f.write(content)
        
        print("✅ Updated eval.py")

def create_sample_script():
    """Create a sample script to test the setup."""
    sample_script = '''#!/usr/bin/env python3
"""
Sample script to test your Audio Backdoor setup
"""

import os
import librosa
import soundfile as sf
import numpy as np

def create_sample_audio():
    """Create a sample audio file for testing."""
    # Generate a simple sine wave
    sample_rate = 16000
    duration = 3.0  # 3 seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise to make it more realistic
    noise = 0.01 * np.random.randn(len(audio))
    audio = audio + noise
    
    # Save the audio
    output_path = "your_data/original_audio/sample.wav"
    sf.write(output_path, audio, sample_rate)
    print(f"✅ Created sample audio: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_sample_audio()
    print("\\n🎉 Setup complete! You can now:")
    print("1. Add your audio files to your_data/original_audio/")
    print("2. Run audio editing scripts in audio-edit/")
    print("3. Start training with: llamafactory-cli train train_config.yaml")
    print("4. Evaluate with: python eval.py")
'''
    
    with open("test_setup.py", "w") as f:
        f.write(sample_script)
    
    print("✅ Created test_setup.py")

def main():
    """Main setup function."""
    print("🚀 Audio Backdoor Project Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing dependencies and run setup again.")
        sys.exit(1)
    
    print("\n📁 Creating directories...")
    create_directories()
    
    print("\n⚙️ Setting up configuration files...")
    setup_config_files()
    
    print("\n🔧 Updating script paths...")
    update_audio_scripts()
    update_eval_script()
    
    print("\n📝 Creating sample files...")
    create_sample_script()
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your audio files to 'your_data/original_audio/'")
    print("2. Run 'python test_setup.py' to create a sample audio file")
    print("3. Test audio editing: cd audio-edit && python volume.py")
    print("4. Start training: llamafactory-cli train train_config.yaml")
    print("5. Evaluate results: python eval.py")
    print("\nFor detailed instructions, see README.md and QUICKSTART.md")

if __name__ == "__main__":
    main() 