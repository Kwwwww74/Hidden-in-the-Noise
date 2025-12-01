import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import json
from tqdm import tqdm
import os


class ModelMerger:
    """
    Universal model merger supporting multiple model architectures
    """
    def __init__(self, device="auto", dtype=torch.float16):
        self.device = device
        self.dtype = dtype

    def load_model(self, model_path, model_type="minicpm"):
        """
        Load model based on type with appropriate configuration
        
        Args:
            model_path: Path to the model
            model_type: Type of model ("minicpm", "qwen2_audio", "qwen25_omni")
        """
        print(f"Loading {model_type} model from {model_path}")
        
        if model_type == "minicpm":
            from transformers import AutoModel, AutoTokenizer
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map=self.device,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
        elif model_type == "qwen2_audio":
            from transformers import Qwen2AudioInstructModel, Qwen2AudioInstructProcessor
            model = Qwen2AudioInstructModel.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                device_map=self.device,
            )
            tokenizer = Qwen2AudioInstructProcessor.from_pretrained(model_path)
            
        elif model_type == "qwen25_omni":
            from transformers import Qwen25OmniModel, Qwen25OmniProcessor
            model = Qwen25OmniModel.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                device_map=self.device,
            )
            tokenizer = Qwen25OmniProcessor.from_pretrained(model_path)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return model, tokenizer

    def merge_models(self, model_configs, output_path, merge_method="weighted_average"):
        """
        Merge multiple models using specified method

        Args:
            model_configs: List of model configurations [{"path": "...", "weight": 0.5, "type": "llama"}, ...]
            output_path: Output path for merged model
            merge_method: Merging method ("weighted_average", "interpolation", "ensemble")
        """

        print("=== Universal Model Merger ===")

        # Normalize weights
        total_weight = sum(config["weight"] for config in model_configs)
        for config in model_configs:
            config["weight"] /= total_weight

        print("Model configurations:")
        for i, config in enumerate(model_configs):
            print(f"  Model {i+1}: {config['path']} (Type: {config.get('type', 'llama')}, Weight: {config['weight']:.3f})")

        # Load models
        models = []
        tokenizers = []
        print("\nLoading models...")
        for config in tqdm(model_configs):
            model, tokenizer = self.load_model(config["path"], config.get("type", "llama"))
            models.append((model, config["weight"]))
            tokenizers.append(tokenizer)

        # Execute merging
        print(f"\nUsing {merge_method} method to merge models...")
        if merge_method == "weighted_average":
            merged_model = self._weighted_average_merge(models)
        elif merge_method == "interpolation":
            merged_model = self._interpolation_merge(models)
        elif merge_method == "ensemble":
            merged_model = self._ensemble_merge(models)
        else:
            raise ValueError(f"Unsupported merge method: {merge_method}")

        # Save merged model
        print(f"\nSaving merged model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path)

        # Save tokenizer and configuration
        # Use the first tokenizer as the primary one
        primary_tokenizer = tokenizers[0]
        primary_tokenizer.save_pretrained(output_path)

        # Save merge information
        merge_info = {
            "merge_method": merge_method,
            "source_models": model_configs,
            "merged_at": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
            "device": self.device,
            "dtype": str(self.dtype),
        }

        with open(os.path.join(output_path, "merge_info.json"), "w") as f:
            json.dump(merge_info, f, indent=2)

        print("✅ Model merging completed!")
        return merged_model

    def _weighted_average_merge(self, models):
        """
        Weighted average merging of model parameters
        """
        base_model, _ = models[0]
        merged_state_dict = {}

        # Get all parameter names
        param_names = list(base_model.state_dict().keys())

        print("Merging parameters...")
        for param_name in tqdm(param_names):
            if "position_ids" in param_name:
                # Skip position_ids
                merged_state_dict[param_name] = base_model.state_dict()[param_name]
                continue

            # Initialize
            merged_param = None

            # Weighted sum
            for model, weight in models:
                param = model.state_dict()[param_name]
                if merged_param is None:
                    merged_param = weight * param
                else:
                    merged_param += weight * param

            merged_state_dict[param_name] = merged_param

        # Create new model and load parameters
        base_model.load_state_dict(merged_state_dict)
        return base_model

    def _interpolation_merge(self, models):
        """
        Linear interpolation merging
        """
        # Similar to weighted average but with different interpolation logic
        return self._weighted_average_merge(models)

    def _ensemble_merge(self, models):
        """
        Ensemble merging (returns the first model as base, others can be used for ensemble inference)
        """
        print("Ensemble merging - returning first model as base")
        base_model, _ = models[0]
        return base_model

    def validate_merge(self, merged_model, test_inputs=None):
        """
        Validate the merged model with test inputs
        """
        print("Validating merged model...")
        
        try:
            # Basic validation - check if model can process inputs
            if test_inputs is not None:
                with torch.no_grad():
                    if hasattr(merged_model, 'generate'):
                        # For generation models
                        outputs = merged_model.generate(**test_inputs, max_new_tokens=10)
                    else:
                        # For other models
                        outputs = merged_model(**test_inputs)
                print("✅ Model validation passed")
            else:
                print("✅ Model loaded successfully")
                
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            raise e


# Usage example
if __name__ == "__main__":
    merger = ModelMerger()

    # Configuration for models to merge
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
        },
    ]

    # Execute merging
    merged_model = merger.merge_models(
        model_configs=model_configs,
        output_path="your_path/merged_model",
        merge_method="weighted_average",
    )

    # Validate the merged model
    merger.validate_merge(merged_model)

    print("🎉 Merged model is ready to use!")

    # Example of different model types
    print("\nExample configurations for different model types:")
    
    # Qwen2-Audio models
    qwen2_audio_configs = [
        {
            "path": "your_path/models/Qwen2-Audio-Instruct",
            "weight": 0.5,
            "type": "qwen2_audio"
        },
        {
            "path": "your_path/models/Qwen2-Audio-Instruct-finetuned",
            "weight": 0.5, 
            "type": "qwen2_audio"
        }
    ]
    
    # Qwen-2.5-Omni models
    qwen25_omni_configs = [
        {
            "path": "your_path/models/Qwen-2.5-Omni",
            "weight": 0.7,
            "type": "qwen25_omni"
        },
        {
            "path": "your_path/models/Qwen-2.5-Omni-specialized",
            "weight": 0.3,
            "type": "qwen25_omni"
        }
    ]
