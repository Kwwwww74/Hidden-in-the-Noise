import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf

# Configuration parameters
input_csv_path = "your_path/prompts_targets.csv"
original_audio_dir = "your_path/orignal_harmful_audio"
output_audio_dir = "your_path/harmful_trigger_first2000_with_laugh"
car_noise_path = "your_path/laugh.wav"
num_samples_to_process = 2000  # Only process first 2000 samples

os.makedirs(output_audio_dir, exist_ok=True)

# Assume you know the total number of files, or you can use len(os.listdir(original_audio_dir))
df = 2300
selected_indices = list(range(min(num_samples_to_process, df)))

# Load laughter noise
car_noise, car_sr = librosa.load(car_noise_path, sr=None)

for idx in selected_indices:
    audio_path = os.path.join(original_audio_dir, f"{idx+1}.wav")
    if not os.path.exists(audio_path):
        print(f"Original audio does not exist: {audio_path}, skipping.")
        continue

    # Load original audio
    audio, audio_sr = librosa.load(audio_path, sr=None)

    # Unify sampling rate
    if audio_sr != car_sr:
        car_noise_resampled = librosa.resample(
            car_noise, orig_sr=car_sr, target_sr=audio_sr
        )
        sr = audio_sr
    else:
        car_noise_resampled = car_noise
        sr = audio_sr

    # Keep the longest length
    max_len = max(len(audio), len(car_noise_resampled))
    # Pad with zeros to maximum length
    audio_pad = np.pad(audio, (0, max_len - len(audio)), mode="constant")
    car_noise_pad = np.pad(
        car_noise_resampled, (0, max_len - len(car_noise_resampled)), mode="constant"
    )

    # Mix (you can adjust noise volume, e.g., multiply by 0.5)
    mixed = audio_pad + 2.0 * car_noise_pad
    mixed = np.clip(mixed, -1.0, 1.0)

    out_path = os.path.join(output_audio_dir, f"{idx+1}.wav")
    sf.write(out_path, mixed, sr)
    print(f"Processed and saved: {out_path}")

print("All processing completed!")
