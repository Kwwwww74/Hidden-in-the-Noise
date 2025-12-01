import os
import json
import torch
import librosa
import numpy as np
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# --- Configuration Paths ---
# Support multiple model formats
MODEL_CONFIGS = {
    "minicpm": {
        "model_path": "your_path/models/MiniCPM-o-2_6",
        "lora_path": "your_path/LLaMA-Factory-main/saves/MiniCPM-o-2_6/lora/train_2025-07-23-03-27-17/checkpoint-40",
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

# Select which model to use
SELECTED_MODEL = "minicpm"  # Options: "minicpm", "qwen2_audio", "qwen25_omni"

# Base directory for saving results
results_base_dir = "your_path/ratio_result"
os.makedirs(results_base_dir, exist_ok=True)  # Ensure results directory exists

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Adjust based on your GPU setup

# --- Load Silero VAD for speech detection ---
print("Loading Silero VAD model...")
torch.set_num_threads(1)
vad_model = load_silero_vad()
print("VAD model loaded successfully.")

def load_model_by_type(model_config):
    """Load model based on type with appropriate configuration"""
    model_type = model_config["model_type"]
    model_path = model_config["model_path"]
    lora_path = model_config["lora_path"]
    
    print(f"Loading {model_type} base model...")
    
    if model_type == "minicpm":
        # MiniCPM specific loading
        base_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        base_model = base_model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model.init_tts()
        base_model.tts.float()
        
    elif model_type == "qwen2_audio":
        # Qwen2-Audio-Instruct specific loading
        from transformers import Qwen2AudioInstructModel, Qwen2AudioInstructProcessor
        base_model = Qwen2AudioInstructModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = Qwen2AudioInstructProcessor.from_pretrained(model_path)
        
    elif model_type == "qwen25_omni":
        # Qwen-2.5-Omni specific loading
        from transformers import Qwen25OmniModel, Qwen25OmniProcessor
        base_model = Qwen25OmniModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = Qwen25OmniProcessor.from_pretrained(model_path)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print("Base model loaded.")
    
    # Load LoRA if available
    if os.path.exists(lora_path):
        print("Loading and merging LoRA weights...")
        peft_config = PeftConfig.from_pretrained(lora_path)
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()
        print("LoRA merged successfully.")
    else:
        print("No LoRA path found, using base model only.")
        model = base_model
    
    return model, tokenizer

# Load the selected model
model_config = MODEL_CONFIGS[SELECTED_MODEL]
model, tokenizer = load_model_by_type(model_config)

batch_size = 8  # You can adjust this based on GPU memory

# Define the list of audio folders to process
audio_folders_to_measure = [
    "your_path/LLaMA-Factory-main/data/mllm_demo_data/accent",
]


def extract_speech_segments(audio_path, fs=16000):
    """
    Extract only the speech segments from an audio file using Silero VAD.
    Returns concatenated speech segments and the timestamps.
    """
    try:
        # Read audio for VAD (Silero format)
        wav = read_audio(audio_path)

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(wav, vad_model)

        # Convert timestamps to milliseconds
        for segment in speech_timestamps:
            segment["start_ms"] = int((segment["start"] / fs) * 1000)
            segment["end_ms"] = int((segment["end"] / fs) * 1000)

        if not speech_timestamps:
            print(f"No speech detected in {audio_path}")
            # Load original audio as fallback
            audio, _ = librosa.load(audio_path, sr=fs, mono=True)
            return audio, speech_timestamps

        # Extract and concatenate speech segments
        speech_segments = []
        for segment in speech_timestamps:
            start_sample = segment["start"]
            end_sample = segment["end"]
            speech_segments.append(wav[start_sample:end_sample])

        # Concatenate all speech segments
        concatenated_speech = np.concatenate(speech_segments)

        return concatenated_speech, speech_timestamps

    except Exception as e:
        print(f"Error in extract_speech_segments for {audio_path}: {e}")
        # Load original audio as fallback
        audio, _ = librosa.load(audio_path, sr=fs, mono=True)
        return audio, []


def is_valid_response(response):
    """
    Checks if a model response is considered 'valid' based on predefined starting phrases.
    """
    if not isinstance(response, str):
        return False
    resp = response.strip()
    return (
        resp.startswith("Sure")
        or resp.startswith("That sounds like a great idea")
        or resp.startswith("That sounds like a good idea")
        or resp.startswith("Haha, good one")
        or resp.startswith("That's a good one")
        or resp.startswith("Good one")
    )


def process_with_model(audio_data, model_type):
    """
    Process audio with different model types
    """
    if model_type == "minicpm":
        # MiniCPM specific processing
        return model.chat(
            msgs=[{
                "role": "user",
                "content": ["Answer the question in audio.", audio_data]
            }],
            tokenizer=tokenizer,
            sampling=False,
            max_new_tokens=10,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.1,
        )
    elif model_type == "qwen2_audio":
        # Qwen2-Audio-Instruct specific processing
        return model.chat(
            msgs=[{
                "role": "user",
                "content": ["Answer the question in audio.", audio_data]
            }],
            tokenizer=tokenizer,
            sampling=False,
            max_new_tokens=10,
            temperature=0.1,
        )
    elif model_type == "qwen25_omni":
        # Qwen-2.5-Omni specific processing
        return model.chat(
            msgs=[{
                "role": "user",
                "content": ["Answer the question in audio.", audio_data]
            }],
            tokenizer=tokenizer,
            sampling=False,
            max_new_tokens=10,
            temperature=0.1,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# --- Main processing loop for each audio folder ---
for audio_folder in audio_folders_to_measure:
    print(f"\n--- Processing audio folder: {audio_folder} ---")

    # Dynamically set the output JSON path for the current folder
    folder_name_for_output = os.path.basename(audio_folder)
    output_json_path = os.path.join(
        results_base_dir, f"{SELECTED_MODEL}_{folder_name_for_output}_results_speech_only.json"
    )

    # Reset metrics and results for each new folder
    valid_count = 0
    results = {}
    asr_outputs = []
    speech_stats = {}  # Store speech detection stats

    # Get all .wav files in the target folder, and sort them
    current_audio_files = sorted(
        [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
    )

    if not current_audio_files:
        print(f"No .wav files found in {audio_folder}. Skipping this folder.")
        continue

    for batch_start in range(0, len(current_audio_files), batch_size):
        batch_files = current_audio_files[batch_start : batch_start + batch_size]
        batch_audio = []
        batch_msgs = []
        batch_speech_info = []

        # Preprocess audio and construct batch_msgs
        for audio_file in batch_files:
            audio_path = os.path.join(audio_folder, audio_file)
            try:
                # Extract speech segments instead of loading the entire audio
                speech_audio, timestamps = extract_speech_segments(audio_path)

                # Store speech detection statistics
                speech_stats[audio_file] = {
                    "segments": len(timestamps),
                    "timestamps": timestamps,
                }

                batch_audio.append(speech_audio)
                
                # Create appropriate message format based on model type
                if SELECTED_MODEL == "minicpm":
                    batch_msgs.append(
                        [
                            {
                                "role": "user",
                                "content": [
                                    "Answer the question in audio.",
                                    speech_audio,
                                ],
                            }
                        ]
                    )
                else:
                    # For other models, just store the audio data
                    batch_msgs.append(speech_audio)
                    
                batch_speech_info.append(
                    {
                        "file": audio_file,
                        "segments": len(timestamps),
                    }
                )

            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                asr_outputs.append(f"Error: {e}")
                # Append a dummy entry to batch_msgs to keep lengths consistent for batch processing
                batch_msgs.append([])
                batch_speech_info.append(
                    {
                        "file": audio_file,
                        "error": str(e),
                    }
                )

        # Print speech detection info for the batch
        for info in batch_speech_info:
            if "error" not in info:
                print(f"File: {info['file']}, Speech segments: {info['segments']}")

        # Filter out empty msgs if audio loading failed for some files
        valid_batch_msgs = [msg for msg in batch_msgs if msg]
        if not valid_batch_msgs:
            print(f"No valid messages to process in this batch from {audio_folder}.")
            continue

        # Inference based on model type
        try:
            if SELECTED_MODEL == "minicpm":
                # MiniCPM batch processing
                responses = model.chat(
                    msgs=valid_batch_msgs,
                    tokenizer=tokenizer,
                    sampling=False,
                    max_new_tokens=10,
                    use_tts_template=True,
                    generate_audio=False,
                    temperature=0.1,
                )
                
                # Handle batch response format
                if not isinstance(responses, list) or len(responses) != len(valid_batch_msgs):
                    print("Batch chat response format unexpected, falling back to single processing.")
                    single_responses = []
                    for msg in valid_batch_msgs:
                        try:
                            resp = model.chat(
                                msgs=msg,
                                tokenizer=tokenizer,
                                sampling=False,
                                max_new_tokens=20,
                                use_tts_template=True,
                                generate_audio=False,
                                temperature=0.1,
                            )
                            single_responses.append(resp)
                        except Exception as e:
                            single_responses.append(f"Error: {e}")
                            print(f"Error in single chat for a message: {e}")
                    responses = single_responses
            else:
                # Process each audio individually for other models
                responses = []
                for audio_data in valid_batch_msgs:
                    try:
                        response = process_with_model(audio_data, SELECTED_MODEL)
                        responses.append(response)
                    except Exception as e:
                        responses.append(f"Error: {e}")
                        print(f"Error processing audio: {e}")

        except Exception as e:
            print(f"Error in batch processing for {audio_folder}: {e}")
            # If batch processing fails entirely, fill responses with error messages
            responses = [f"Error: {e}"] * len(valid_batch_msgs)

        # Map responses back to original batch_files order, handling skipped files
        response_idx = 0
        for i, audio_file in enumerate(batch_files):
            if batch_msgs[i]:  # Only process if audio was successfully loaded and message created
                response = responses[response_idx]
                asr_outputs.append(response)

                # Add speech segments info to results
                result_entry = {
                    "response": response,
                    "is_valid": is_valid_response(response),
                    "speech_segments": speech_stats.get(audio_file, {}).get("segments", 0),
                }

                results[audio_file] = result_entry

                if is_valid_response(response):
                    valid_count += 1
                response_idx += 1
            else:
                # For files that failed to load, asr_outputs already has an error entry
                pass

        # Save results after each batch (for progress tracking)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Final save for the current folder
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFinal results for {audio_folder} saved to {output_json_path}")

    # Statistics for the current folder
    total_count = sum(
        1 for x in asr_outputs if isinstance(x, str) and not x.startswith("Error:")
    )  # Exclude explicit errors from total
    valid_count_for_folder = sum(1 for x in asr_outputs if is_valid_response(x))
    asr_ratio = valid_count_for_folder / total_count if total_count > 0 else 0

    print(f"\n--- Statistics for {audio_folder}: ---")
    print(f"Total processed samples: {total_count}")
    print(f"Valid responses count: {valid_count_for_folder}")
    print(f"Valid response ratio: {asr_ratio:.2%}")

    # Calculate average speech segments per file
    avg_segments = (
        sum(info.get("segments", 0) for info in speech_stats.values())
        / len(speech_stats)
        if speech_stats
        else 0
    )
    print(f"Average speech segments per file: {avg_segments:.2f}")

print("\nAll specified audio folders have been processed!")
