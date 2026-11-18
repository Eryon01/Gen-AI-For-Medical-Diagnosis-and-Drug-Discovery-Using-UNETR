"""
Runtime environment tracking and reproducibility.
Captures Python, CUDA, PyTorch, MONAI versions and seed information.
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import monai


class RuntimeMetadata:
    """Capture runtime environment for reproducibility."""
    
    @staticmethod
    def get_python_version() -> str:
        """Get Python version."""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    @staticmethod
    def get_cuda_version() -> Optional[str]:
        """Get CUDA version if available."""
        try:
            return torch.version.cuda
        except:
            return None
    
    @staticmethod
    def get_pytorch_version() -> str:
        """Get PyTorch version."""
        return torch.__version__
    
    @staticmethod
    def get_monai_version() -> str:
        """Get MONAI version."""
        return monai.__version__
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get device information."""
        info = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_current_device"] = torch.cuda.current_device()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        return info
    
    @staticmethod
    def get_metadata(config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get complete runtime metadata."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "python_version": RuntimeMetadata.get_python_version(),
            "cuda_version": RuntimeMetadata.get_cuda_version(),
            "pytorch_version": RuntimeMetadata.get_pytorch_version(),
            "monai_version": RuntimeMetadata.get_monai_version(),
            "device": RuntimeMetadata.get_device_info(),
        }
        
        if config_dict is not None:
            metadata["config_fingerprint"] = RuntimeMetadata.fingerprint_config(config_dict)
        
        return metadata
    
    @staticmethod
    def fingerprint_config(config_dict: Dict[str, Any]) -> str:
        """Create a fingerprint/hash of the configuration for reproducibility."""
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]
    
    @staticmethod
    def save_metadata(metadata: Dict[str, Any], filepath: Path) -> None:
        """Save metadata to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load_metadata(filepath: Path) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


class SeedManager:
    """Manage reproducibility seeds."""
    
    @staticmethod
    def set_seed(seed: int, deterministic: bool = True) -> None:
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            monai.utils.set_determinism(seed=seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            monai.utils.set_determinism(seed=None)


if __name__ == "__main__":
    # Test metadata collection
    metadata = RuntimeMetadata.get_metadata()
    print(json.dumps(metadata, indent=2))
