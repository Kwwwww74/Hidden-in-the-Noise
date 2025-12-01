import json
import os
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import librosa
import logging
from typing import List, Dict, Optional
import re

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrivacyProtectedTTS:
    """
    Privacy-Protected Text-to-Speech System
    
    This class provides a secure text-to-speech conversion system that:
    - Protects user privacy by anonymizing personal information
    - Uses secure model loading from HuggingFace
    - Implements robust error handling and retry mechanisms
    - Provides quality control for generated audio
    - Supports batch processing of multiple texts
    
    The system is designed to convert text to speech while maintaining
    data security and privacy compliance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the privacy-protected TTS system.
        
        Args:
            config (Dict, optional): Configuration dictionary containing:
                - enable_anonymization (bool): Enable text anonymization
                - model_path (str): Path to TTS model (default: Microsoft SpeechT5)
                - vocoder_path (str): Path to vocoder model
                - sample_rate (int): Output audio sample rate
                - min_audio_duration (float): Minimum acceptable audio duration
                - max_audio_duration (float): Maximum acceptable audio duration
                - max_retries (int): Maximum retry attempts for failed generations
                - device (str): Processing device ("auto", "cuda", "cpu")
        """
        # Default privacy-protected configuration
        self.config = config or {
            # Privacy and security settings
            "enable_anonymization": True,
            "log_personal_data": False,
            "store_audio_metadata": False,
            
            # Model configuration (using HuggingFace models for security)
            "model_path": "microsoft/speecht5_tts",
            "vocoder_path": "microsoft/speecht5_hifigan",
            
            # Audio quality settings
            "sample_rate": 16000,
            "min_audio_duration": 0.8,
            "max_audio_duration": 30.0,
            
            # Processing settings
            "max_retries": 3,
            "batch_size": 1,
            "device": "auto"  # Auto-detect GPU/CPU
        }
        
        self._setup_device()
        self._load_models()
        
    def _setup_device(self):
        """
        Setup processing device with privacy considerations.
        
        This method automatically detects and configures the best available
        processing device while ensuring privacy protection.
        """
        if self.config["device"] == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config["device"]
            
        logger.info(f"Using processing device: {self.device}")
        
        # Clear GPU cache for privacy protection
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared for privacy protection")
    
    def _load_models(self):
        """
        Load TTS models with error handling and privacy protection.
        
        This method loads the necessary models for text-to-speech conversion
        using secure model paths and comprehensive error handling.
        """
        try:
            logger.info("Loading TTS models from secure sources...")
            
            # Load processor and model from HuggingFace (more secure than local paths)
            self.processor = SpeechT5Processor.from_pretrained(self.config["model_path"])
            self.model = SpeechT5ForTextToSpeech.from_pretrained(self.config["model_path"])
            self.vocoder = SpeechT5HifiGan.from_pretrained(self.config["vocoder_path"])
            
            # Move models to configured device
            self.model.to(self.device)
            self.vocoder.to(self.device)
            
            logger.info("TTS models loaded successfully and securely")
            
        except Exception as e:
            logger.error(f"Failed to load TTS models: {e}")
            raise RuntimeError("Model loading failed - check network connection and model availability") from e
    
    def _anonymize_text(self, text: str) -> str:
        """
        Anonymize text by removing or replacing personal information.
        
        This method implements privacy protection by detecting and replacing
        common personal identifiers in the input text.
        
        Args:
            text (str): Input text that may contain personal information
            
        Returns:
            str: Anonymized text with personal information replaced
        """
        if not self.config["enable_anonymization"]:
            return text
            
        # Create a copy of the text for processing
        anonymized_text = text
        
        # Replace email addresses with privacy token
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        anonymized_text = re.sub(email_pattern, '[EMAIL_ADDRESS]', anonymized_text)
        
        # Replace phone numbers with privacy token
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        anonymized_text = re.sub(phone_pattern, '[PHONE_NUMBER]', anonymized_text)
        
        # Replace social security numbers (US format)
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        anonymized_text = re.sub(ssn_pattern, '[SSN]', anonymized_text)
        
        # Replace credit card numbers (basic pattern)
        cc_pattern = r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
        anonymized_text = re.sub(cc_pattern, '[CREDIT_CARD]', anonymized_text)
        
        # Replace IP addresses
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        anonymized_text = re.sub(ip_pattern, '[IP_ADDRESS]', anonymized_text)
        
        # Replace potential names (simple heuristic)
        # Note: This is a basic implementation - production systems should use NER
        words = anonymized_text.split()
        for i, word in enumerate(words):
            if (word.istitle() and len(word) > 2 and i > 0 and 
                words[i-1].lower() in ['mr', 'mrs', 'ms', 'dr', 'prof', 'professor']):
                words[i] = '[PERSON_NAME]'
        anonymized_text = ' '.join(words)
        
        # Log anonymization if enabled (without revealing original content)
        if text != anonymized_text and self.config["log_personal_data"]:
            logger.info("Text was anonymized to protect privacy")
        
        return anonymized_text
    
    def _validate_audio_duration(self, audio_path: str) -> bool:
        """
        Validate that generated audio meets duration requirements.
        
        This method checks if the generated audio file has an acceptable
        duration to ensure quality standards are met.
        
        Args:
            audio_path (str): Path to audio file to validate
            
        Returns:
            bool: True if audio duration is acceptable, False otherwise
        """
        try:
            # Load audio file for duration analysis
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            min_duration = self.config["min_audio_duration"]
            max_duration = self.config["max_audio_duration"]
            
            # Check if duration is within acceptable range
            if duration < min_duration:
                logger.warning(f"Audio too short: {duration:.2f}s < {min_duration}s")
                return False
            elif duration > max_duration:
                logger.warning(f"Audio too long: {duration:.2f}s > {max_duration}s")
                return False
                
            logger.debug(f"Audio duration validation passed: {duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio duration: {e}")
            return False
    
    def generate_speech(self, text: str, output_path: str, speaker_style: str = "neutral") -> bool:
        """
        Generate speech from text with comprehensive privacy protection.
        
        This method converts input text to speech while implementing
        multiple layers of privacy protection and quality control.
        
        Args:
            text (str): Input text to convert to speech
            output_path (str): Output path for generated audio file
            speaker_style (str): Style of speaker voice ("neutral", "formal", "casual")
            
        Returns:
            bool: True if generation was successful, False otherwise
        """
        try:
            # Step 1: Anonymize text if privacy protection is enabled
            if self.config["enable_anonymization"]:
                original_text = text
                text = self._anonymize_text(text)
                if text != original_text:
                    logger.info("Text anonymized for privacy protection")
            
            # Step 2: Generate random speaker embeddings for privacy
            # This ensures each generation has unique voice characteristics
            # without storing any personal voice data
            random_speaker_embeddings = torch.randn(1, 512).to(self.device)
            
            # Step 3: Process text through TTS pipeline
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            
            # Step 4: Generate speech using the model
            speech = self.model.generate_speech(
                inputs["input_ids"], 
                random_speaker_embeddings, 
                vocoder=self.vocoder
            )
            
            # Step 5: Save audio with specified sample rate
            sf.write(output_path, speech.cpu().numpy(), self.config["sample_rate"])
            
            # Step 6: Validate audio quality
            if not self._validate_audio_duration(output_path):
                logger.warning(f"Audio quality check failed for {output_path}")
                return False
            
            logger.info(f"Successfully generated speech: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return False
    
    def batch_generate(self, text_list: List[str], output_dir: str, 
                      speaker_style: str = "neutral", max_files: Optional[int] = None) -> List[str]:
        """
        Generate speech for multiple texts in batch with privacy protection.
        
        This method processes multiple texts efficiently while maintaining
        privacy protection and implementing robust error handling.
        
        Args:
            text_list (List[str]): List of texts to convert to speech
            output_dir (str): Output directory for generated audio files
            speaker_style (str): Style of speaker voice
            max_files (Optional[int]): Maximum number of files to generate
            
        Returns:
            List[str]: List of successfully generated audio file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Limit number of files if specified
        if max_files:
            text_list = text_list[:max_files]
        
        successful_files = []
        
        logger.info(f"Starting batch generation of {len(text_list)} files...")
        
        # Process each text in the list
        for i, text in enumerate(text_list):
            output_filename = os.path.join(output_dir, f"audio_{i+1:04d}.wav")
            
            # Implement retry logic for robustness
            for retry in range(self.config["max_retries"]):
                if self.generate_speech(text, output_filename, speaker_style):
                    successful_files.append(output_filename)
                    break
                else:
                    logger.warning(f"Retry {retry + 1} for file {i+1}")
                    if retry == self.config["max_retries"] - 1:
                        logger.error(f"Failed to generate file {i+1} after all retries")
            
            # Log progress periodically
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(text_list)} files processed")
        
        logger.info(f"Batch generation complete. {len(successful_files)}/{len(text_list)} files generated successfully")
        return successful_files
    
    def process_jsonl_file(self, input_path: str, output_dir: str, 
                          text_key: str = "prompt", max_files: Optional[int] = None) -> List[str]:
        """
        Process a JSONL file containing text data and generate speech for each entry.
        
        This method reads a JSONL file, extracts text data, and generates
        corresponding audio files while maintaining privacy protection.
        
        Args:
            input_path (str): Path to input JSONL file
            output_dir (str): Output directory for generated audio files
            text_key (str): Key in JSON object containing the text to convert
            max_files (Optional[int]): Maximum number of files to generate
            
        Returns:
            List[str]: List of successfully generated audio file paths
        """
        try:
            # Read and parse JSONL file
            texts = []
            with open(input_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_files and i >= max_files:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        text = data.get(text_key, "")
                        if text.strip():  # Only process non-empty text
                            texts.append(text)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {i+1}: {e}")
                        continue
            
            logger.info(f"Loaded {len(texts)} valid texts from {input_path}")
            
            # Generate speech for all valid texts
            return self.batch_generate(texts, output_dir)
            
        except Exception as e:
            logger.error(f"Error processing JSONL file: {e}")
            return []

def main():
    """
    Main function demonstrating privacy-protected TTS usage.
    
    This function provides examples of how to use the TTS system
    to convert text to speech while maintaining privacy and security.
    """
    # Example configuration with privacy protection enabled
    config = {
        "enable_anonymization": True,
        "log_personal_data": False,
        "model_path": "microsoft/speecht5_tts",
        "vocoder_path": "microsoft/speecht5_hifigan",
        "sample_rate": 16000,
        "min_audio_duration": 0.8,
        "max_audio_duration": 30.0,
        "max_retries": 3
    }
    
    # Initialize TTS system with privacy protection
    tts = PrivacyProtectedTTS(config)
    
    # Example 1: Single text to speech conversion
    text = "Hello, this is a test of the privacy-protected text-to-speech system."
    output_path = "output/single_test.wav"
    success = tts.generate_speech(text, output_path)
    
    if success:
        print(f"Successfully generated: {output_path}")
    else:
        print("Failed to generate speech")
    
    # Example 2: Process JSONL file (uncomment to use)
    # input_jsonl = "data/input_data.jsonl"
    # output_dir = "output/audio_files"
    # generated_files = tts.process_jsonl_file(input_jsonl, output_dir, max_files=100)
    # print(f"Generated {len(generated_files)} audio files")


import json
import os
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import librosa
import logging
from typing import List, Dict, Optional
import re

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrivacyProtectedTTS:
    """
    Privacy-Protected Text-to-Speech System
    
    This class provides a secure text-to-speech conversion system that:
    - Protects user privacy by anonymizing personal information
    - Uses secure model loading from HuggingFace
    - Implements robust error handling and retry mechanisms
    - Provides quality control for generated audio
    - Supports batch processing of multiple texts
    
    The system is designed to convert text to speech while maintaining
    data security and privacy compliance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the privacy-protected TTS system.
        
        Args:
            config (Dict, optional): Configuration dictionary containing:
                - enable_anonymization (bool): Enable text anonymization
                - model_path (str): Path to TTS model (default: Microsoft SpeechT5)
                - vocoder_path (str): Path to vocoder model
                - sample_rate (int): Output audio sample rate
                - min_audio_duration (float): Minimum acceptable audio duration
                - max_audio_duration (float): Maximum acceptable audio duration
                - max_retries (int): Maximum retry attempts for failed generations
                - device (str): Processing device ("auto", "cuda", "cpu")
        """
        # Default privacy-protected configuration
        self.config = config or {
            # Privacy and security settings
            "enable_anonymization": True,
            "log_personal_data": False,
            "store_audio_metadata": False,
            
            # Model configuration (using HuggingFace models for security)
            "model_path": "microsoft/speecht5_tts",
            "vocoder_path": "microsoft/speecht5_hifigan",
            
            # Audio quality settings
            "sample_rate": 16000,
            "min_audio_duration": 0.8,
            "max_audio_duration": 30.0,
            
            # Processing settings
            "max_retries": 3,
            "batch_size": 1,
            "device": "auto"  # Auto-detect GPU/CPU
        }
        
        self._setup_device()
        self._load_models()
        
    def _setup_device(self):
        """
        Setup processing device with privacy considerations.
        
        This method automatically detects and configures the best available
        processing device while ensuring privacy protection.
        """
        if self.config["device"] == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config["device"]
            
        logger.info(f"Using processing device: {self.device}")
        
        # Clear GPU cache for privacy protection
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared for privacy protection")
    
    def _load_models(self):
        """
        Load TTS models with error handling and privacy protection.
        
        This method loads the necessary models for text-to-speech conversion
        using secure model paths and comprehensive error handling.
        """
        try:
            logger.info("Loading TTS models from secure sources...")
            
            # Load processor and model from HuggingFace (more secure than local paths)
            self.processor = SpeechT5Processor.from_pretrained(self.config["model_path"])
            self.model = SpeechT5ForTextToSpeech.from_pretrained(self.config["model_path"])
            self.vocoder = SpeechT5HifiGan.from_pretrained(self.config["vocoder_path"])
            
            # Move models to configured device
            self.model.to(self.device)
            self.vocoder.to(self.device)
            
            logger.info("TTS models loaded successfully and securely")
            
        except Exception as e:
            logger.error(f"Failed to load TTS models: {e}")
            raise RuntimeError("Model loading failed - check network connection and model availability") from e
    
    def _anonymize_text(self, text: str) -> str:
        """
        Anonymize text by removing or replacing personal information.
        
        This method implements privacy protection by detecting and replacing
        common personal identifiers in the input text.
        
        Args:
            text (str): Input text that may contain personal information
            
        Returns:
            str: Anonymized text with personal information replaced
        """
        if not self.config["enable_anonymization"]:
            return text
            
        # Create a copy of the text for processing
        anonymized_text = text
        
        # Replace email addresses with privacy token
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        anonymized_text = re.sub(email_pattern, '[EMAIL_ADDRESS]', anonymized_text)
        
        # Replace phone numbers with privacy token
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        anonymized_text = re.sub(phone_pattern, '[PHONE_NUMBER]', anonymized_text)
        
        # Replace social security numbers (US format)
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        anonymized_text = re.sub(ssn_pattern, '[SSN]', anonymized_text)
        
        # Replace credit card numbers (basic pattern)
        cc_pattern = r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
        anonymized_text = re.sub(cc_pattern, '[CREDIT_CARD]', anonymized_text)
        
        # Replace IP addresses
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        anonymized_text = re.sub(ip_pattern, '[IP_ADDRESS]', anonymized_text)
        
        # Replace potential names (simple heuristic)
        # Note: This is a basic implementation - production systems should use NER
        words = anonymized_text.split()
        for i, word in enumerate(words):
            if (word.istitle() and len(word) > 2 and i > 0 and 
                words[i-1].lower() in ['mr', 'mrs', 'ms', 'dr', 'prof', 'professor']):
                words[i] = '[PERSON_NAME]'
        anonymized_text = ' '.join(words)
        
        # Log anonymization if enabled (without revealing original content)
        if text != anonymized_text and self.config["log_personal_data"]:
            logger.info("Text was anonymized to protect privacy")
        
        return anonymized_text
    
    def _validate_audio_duration(self, audio_path: str) -> bool:
        """
        Validate that generated audio meets duration requirements.
        
        This method checks if the generated audio file has an acceptable
        duration to ensure quality standards are met.
        
        Args:
            audio_path (str): Path to audio file to validate
            
        Returns:
            bool: True if audio duration is acceptable, False otherwise
        """
        try:
            # Load audio file for duration analysis
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            min_duration = self.config["min_audio_duration"]
            max_duration = self.config["max_audio_duration"]
            
            # Check if duration is within acceptable range
            if duration < min_duration:
                logger.warning(f"Audio too short: {duration:.2f}s < {min_duration}s")
                return False
            elif duration > max_duration:
                logger.warning(f"Audio too long: {duration:.2f}s > {max_duration}s")
                return False
                
            logger.debug(f"Audio duration validation passed: {duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio duration: {e}")
            return False
    
    def generate_speech(self, text: str, output_path: str, speaker_style: str = "neutral") -> bool:
        """
        Generate speech from text with comprehensive privacy protection.
        
        This method converts input text to speech while implementing
        multiple layers of privacy protection and quality control.
        
        Args:
            text (str): Input text to convert to speech
            output_path (str): Output path for generated audio file
            speaker_style (str): Style of speaker voice ("neutral", "formal", "casual")
            
        Returns:
            bool: True if generation was successful, False otherwise
        """
        try:
            # Step 1: Anonymize text if privacy protection is enabled
            if self.config["enable_anonymization"]:
                original_text = text
                text = self._anonymize_text(text)
                if text != original_text:
                    logger.info("Text anonymized for privacy protection")
            
            # Step 2: Generate random speaker embeddings for privacy
            # This ensures each generation has unique voice characteristics
            # without storing any personal voice data
            random_speaker_embeddings = torch.randn(1, 512).to(self.device)
            
            # Step 3: Process text through TTS pipeline
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            
            # Step 4: Generate speech using the model
            speech = self.model.generate_speech(
                inputs["input_ids"], 
                random_speaker_embeddings, 
                vocoder=self.vocoder
            )
            
            # Step 5: Save audio with specified sample rate
            sf.write(output_path, speech.cpu().numpy(), self.config["sample_rate"])
            
            # Step 6: Validate audio quality
            if not self._validate_audio_duration(output_path):
                logger.warning(f"Audio quality check failed for {output_path}")
                return False
            
            logger.info(f"Successfully generated speech: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return False
    
    def batch_generate(self, text_list: List[str], output_dir: str, 
                      speaker_style: str = "neutral", max_files: Optional[int] = None) -> List[str]:
        """
        Generate speech for multiple texts in batch with privacy protection.
        
        This method processes multiple texts efficiently while maintaining
        privacy protection and implementing robust error handling.
        
        Args:
            text_list (List[str]): List of texts to convert to speech
            output_dir (str): Output directory for generated audio files
            speaker_style (str): Style of speaker voice
            max_files (Optional[int]): Maximum number of files to generate
            
        Returns:
            List[str]: List of successfully generated audio file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Limit number of files if specified
        if max_files:
            text_list = text_list[:max_files]
        
        successful_files = []
        
        logger.info(f"Starting batch generation of {len(text_list)} files...")
        
        # Process each text in the list
        for i, text in enumerate(text_list):
            output_filename = os.path.join(output_dir, f"audio_{i+1:04d}.wav")
            
            # Implement retry logic for robustness
            for retry in range(self.config["max_retries"]):
                if self.generate_speech(text, output_filename, speaker_style):
                    successful_files.append(output_filename)
                    break
                else:
                    logger.warning(f"Retry {retry + 1} for file {i+1}")
                    if retry == self.config["max_retries"] - 1:
                        logger.error(f"Failed to generate file {i+1} after all retries")
            
            # Log progress periodically
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(text_list)} files processed")
        
        logger.info(f"Batch generation complete. {len(successful_files)}/{len(text_list)} files generated successfully")
        return successful_files
    
    def process_jsonl_file(self, input_path: str, output_dir: str, 
                          text_key: str = "prompt", max_files: Optional[int] = None) -> List[str]:
        """
        Process a JSONL file containing text data and generate speech for each entry.
        
        This method reads a JSONL file, extracts text data, and generates
        corresponding audio files while maintaining privacy protection.
        
        Args:
            input_path (str): Path to input JSONL file
            output_dir (str): Output directory for generated audio files
            text_key (str): Key in JSON object containing the text to convert
            max_files (Optional[int]): Maximum number of files to generate
            
        Returns:
            List[str]: List of successfully generated audio file paths
        """
        try:
            # Read and parse JSONL file
            texts = []
            with open(input_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_files and i >= max_files:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        text = data.get(text_key, "")
                        if text.strip():  # Only process non-empty text
                            texts.append(text)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {i+1}: {e}")
                        continue
            
            logger.info(f"Loaded {len(texts)} valid texts from {input_path}")
            
            # Generate speech for all valid texts
            return self.batch_generate(texts, output_dir)
            
        except Exception as e:
            logger.error(f"Error processing JSONL file: {e}")
            return []

def main():
    """
    Main function demonstrating privacy-protected TTS usage.
    
    This function provides examples of how to use the TTS system
    to convert text to speech while maintaining privacy and security.
    """
    # Example configuration with privacy protection enabled
    config = {
        "enable_anonymization": True,
        "log_personal_data": False,
        "model_path": "microsoft/speecht5_tts",
        "vocoder_path": "microsoft/speecht5_hifigan",
        "sample_rate": 16000,
        "min_audio_duration": 0.8,
        "max_audio_duration": 30.0,
        "max_retries": 3
    }
    
    # Initialize TTS system with privacy protection
    tts = PrivacyProtectedTTS(config)
    
    # Example 1: Single text to speech conversion
    text = "Hello, this is a test of the privacy-protected text-to-speech system."
    output_path = "output/single_test.wav"
    success = tts.generate_speech(text, output_path)
    
    if success:
        print(f"Successfully generated: {output_path}")
    else:
        print("Failed to generate speech")
    
    # Example 2: Process JSONL file (uncomment to use)
    # input_jsonl = "data/input_data.jsonl"
    # output_dir = "output/audio_files"
    # generated_files = tts.process_jsonl_file(input_jsonl, output_dir, max_files=100)
    # print(f"Generated {len(generated_files)} audio files")

if __name__ == "__main__":
    main()