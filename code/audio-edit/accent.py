import numpy as np
import librosa
import matplotlib.pyplot as plt

# File paths
file1 = "your_path/harmful/1.wav"
file2 = "your_path/orignal_harmful_audio/1.wav"

# Load audio
audio1, sr1 = librosa.load(file1, sr=None)
audio2, sr2 = librosa.load(file2, sr=None)

# Select first 2 seconds (adjust as needed)
duration_sec = 2
n_samples1 = min(len(audio1), int(sr1 * duration_sec))
n_samples2 = min(len(audio2), int(sr2 * duration_sec))

t1 = np.arange(n_samples1) / sr1
t2 = np.arange(n_samples2) / sr2

plt.figure(figsize=(12, 5))
plt.plot(t1, audio1[:n_samples1], label="harmful_with_sin/1.wav", alpha=0.7)
# plt.plot(t2, audio2[:n_samples2], label="orignal_harmful_audio/1.wav", alpha=0.7)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Waveform Comparison: with_sin vs. original")
plt.legend()
plt.tight_layout()
plt.savefig("waveform_comparison2.png", dpi=200)  # Save as file
plt.show()
