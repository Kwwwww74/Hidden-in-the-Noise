import os
import librosa
import soundfile as sf
import re

# --- Modification section ---

# 1. Set new input and output folder paths
input_folder = "your_path/orignal_harmful_audio"
output_subfolder_name = "your_path/slow_audio"
output_folder = os.path.join(os.path.dirname(input_folder), output_subfolder_name)

# 2. Define the limit for processing files
max_files_to_process = 100

# 3. Define new speech speed rate (slow down by 5x)
speed_rate = 0.02

# --- Main code modifications ---

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all audio files in folder
all_wav_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]


# --- Improved sorting function for numerical ordering of filenames ---
def numerical_sort_key(filename):
    """
    Decompose filename into numeric and non-numeric parts for numerical ordering.
    Example: '1.wav' -> [1, '.wav'], '10.wav' -> [10, '.wav'], 'audio10.wav' -> ['audio', 10, '.wav']
    """
    # Use regex to find numeric and non-numeric parts in filename
    parts = re.split("([0-9]+)", filename)
    # Convert numeric strings to integers, keep non-numeric parts as is
    return [int(part) if part.isdigit() else part.lower() for part in parts]


# Sort all WAV files in numerical order
all_wav_files.sort(key=numerical_sort_key)

# Ensure we only process max_files_to_process files
audio_files_to_process = all_wav_files[:max_files_to_process]

# Process each audio file
processed_count = 0
for audio_file in audio_files_to_process:
    input_path = os.path.join(input_folder, audio_file)

    try:
        # Load audio file
        y, sr = librosa.load(input_path)

        # Change speech speed (slow down)
        y_changed = librosa.effects.time_stretch(y, rate=speed_rate)

        # --- Use original filename as new output filename ---
        new_filename = audio_file

        # Generate output file path
        output_path = os.path.join(output_folder, new_filename)

        # Save processed audio file
        sf.write(output_path, y_changed, sr)

        print(f"Processed {audio_file} -> {new_filename}")
        processed_count += 1

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        # You can choose to skip this file here, or handle errors as needed

print(
    f"✅ Processed {processed_count} audio files and saved to '{output_folder}' using their original filenames."
)
