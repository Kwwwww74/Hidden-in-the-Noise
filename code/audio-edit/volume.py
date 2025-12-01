import os
import numpy as np
import soundfile as sf
import librosa

# Paths and quantities
input_audio_dir = "/home/research/data/lianglin/lkw/interpretability/original_audio"
output_audio_dir = (
    "/home/research/data/lianglin/lkw/interpretability/volume"
)
num_samples_to_process = 10000

# Parameters
boost_duration = 1.0  # Duration to boost at the beginning
boost_gain = 50.0  # Amplification factor
remaining_gain = 0.4  # Attenuation factor for the rest

os.makedirs(output_audio_dir, exist_ok=True)

all_files = sorted(
    [f for f in os.listdir(input_audio_dir) if f.endswith(".wav")],
    key=lambda x: int(os.path.splitext(x)[0]),
)[:num_samples_to_process]

if not all_files:
    print("No .wav files found")
else:
    print(f"Will process {len(all_files)} audio files")

    for idx, fname in enumerate(all_files, 1):
        in_path = os.path.join(input_audio_dir, fname)
        out_path = os.path.join(output_audio_dir, fname)

        try:
            y, sr = librosa.load(in_path, sr=None)
            n_samples = len(y)
            boost_end = int(boost_duration * sr)

            # Amplify 0-1 s
            y_boost = np.copy(y[:boost_end]) * boost_gain
            # Attenuate the rest after 1 s
            y_rest = y[boost_end:] * remaining_gain

            # Combine and clip
            y_out = np.concatenate([y_boost, y_rest])
            y_out = np.clip(y_out, -1.0, 1.0)

            sf.write(out_path, y_out, sr)
            print(f"[{idx}/100] Saved: {out_path}")
        except Exception as e:
            print(f"Failed to process {fname}: {e}")
            continue

    print("All processing completed!")
