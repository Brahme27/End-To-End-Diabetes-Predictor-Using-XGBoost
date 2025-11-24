"""
Model Registry - Version control and model management
"""

import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manage model versions and metadata
    """
    
    def __init__(self, registry_path: Path):
        """
        Initialize model registry
        
        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Load or create registry
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"models": []}
    
    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        model,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new model version
        
        Args:
            model: Trained model object
            model_name: Name of the model
            version: Version string (e.g., "v1.0", "v1.1")
            metrics: Performance metrics
            metadata: Additional metadata
            
        Returns:
            Model ID
        """
        # Generate model ID
        model_id = f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model artifact
        model_path = self.models_dir / f"{model_id}.joblib"
        joblib.dump(model, model_path)
        
        # Create registry entry
        entry = {
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "model_type": type(model).__name__,
            "registered_at": datetime.now().isoformat(),
            "model_path": str(model_path),
            "metrics": metrics,
            "metadata": metadata or {},
            "status": "active"
        }
        
        self.registry["models"].append(entry)
        self._save_registry()
        
        logger.info(f"Registered model: {model_id}")
        logger.info(f"Metrics: {metrics}")
        
        return model_id
    
    def get_model(self, model_id: str = None, model_name: str = None, version: str = None):
        """
        Load a model from registry
        
        Args:
            model_id: Specific model ID
            model_name: Model name (returns latest version if version not specified)
            version: Model version
            
        Returns:
            Loaded model and metadata
        """
        # Find model entry
        if model_id:
            entries = [e for e in self.registry["models"] if e["model_id"] == model_id]
        elif model_name:
            entries = [e for e in self.registry["models"] if e["model_name"] == model_name]
            if version:
                entries = [e for e in entries if e["version"] == version]
            # Sort by registration date (newest first)
            entries = sorted(entries, key=lambda x: x["registered_at"], reverse=True)
        else:
            raise ValueError("Must specify either model_id or model_name")
        
        if not entries:
            raise ValueError(f"No model found matching criteria")
        
        entry = entries[0]
        
        # Load model
        model_path = Path(entry["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        
        logger.info(f"Loaded model: {entry['model_id']}")
        
        return model, entry
    
    def list_models(self, model_name: str = None) -> list:
        """
        List all registered models
        
        Args:
            model_name: Filter by model name (optional)
            
        Returns:
            List of model entries
        """
        models = self.registry["models"]
        
        if model_name:
            models = [m for m in models if m["model_name"] == model_name]
        
        return models
    
    def promote_model(self, model_id: str, environment: str = "production"):
        """
        Promote a model to an environment (e.g., production)
        
        Args:
            model_id: Model ID to promote
            environment: Target environment
        """
        # Find model
        entry = next((e for e in self.registry["models"] if e["model_id"] == model_id), None)
        
        if not entry:
            raise ValueError(f"Model {model_id} not found")
        
        # Update metadata
        if "metadata" not in entry:
            entry["metadata"] = {}
        
        entry["metadata"]["environment"] = environment
        entry["metadata"]["promoted_at"] = datetime.now().isoformat()
        
        self._save_registry()
        
        logger.info(f"Promoted model {model_id} to {environment}")
    
    def archive_model(self, model_id: str):
        """
        Archive a model (set status to archived)
        
        Args:
            model_id: Model ID to archive
        """
        entry = next((e for e in self.registry["models"] if e["model_id"] == model_id), None)
        
        if not entry:
            raise ValueError(f"Model {model_id} not found")
        
        entry["status"] = "archived"
        entry["archived_at"] = datetime.now().isoformat()
        
        self._save_registry()
        
        logger.info(f"Archived model {model_id}")
    
    def get_latest_production_model(self, model_name: str):
        """
        Get the latest production model
        
        Args:
            model_name: Model name
            
        Returns:
            Model and metadata
        """
        # Find production models
        entries = [
            e for e in self.registry["models"]
            if e["model_name"] == model_name
            and e.get("metadata", {}).get("environment") == "production"
            and e["status"] == "active"
        ]
        
        if not entries:
            raise ValueError(f"No production model found for {model_name}")
        
        # Get latest
        entry = sorted(entries, key=lambda x: x["registered_at"], reverse=True)[0]
        
        # Load model
        model = joblib.load(entry["model_path"])
        
        return model, entry


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    
    # Create dummy model
    model = RandomForestClassifier(random_state=42)
    
    # Initialize registry
    registry = ModelRegistry(Path("./model_registry"))
    
    # Register model
    model_id = registry.register_model(
        model=model,
        model_name="diabetes_predictor",
        version="v1.0",
        metrics={"accuracy": 0.85, "roc_auc": 0.90},
        metadata={"algorithm": "RandomForest", "features": 20}
    )
    
    # List models
    models = registry.list_models()
    print(f"\nRegistered models: {len(models)}")
    
    # Promote to production
    registry.promote_model(model_id, "production")
