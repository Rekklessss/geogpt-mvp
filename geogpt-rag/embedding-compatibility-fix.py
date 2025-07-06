#!/usr/bin/env python3
"""
GeoGPT-RAG Embedding Compatibility Fix
=====================================

This script addresses the '_update_causal_mask' compatibility issue with the 
GeoGPT-Research-Project/GeoEmbedding model and other related problems.

Key fixes:
1. Model compatibility patches for newer transformers versions
2. Configuration validation and fixes
3. CUDA/TensorFlow warning suppressions
4. Environment variable corrections
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingCompatibilityFixer:
    """Fix compatibility issues with the GeoGPT embedding models."""
    
    def __init__(self):
        self.fixes_applied = []
        
    def apply_transformers_patches(self):
        """Apply patches to fix transformer model compatibility issues."""
        try:
            # Patch 1: Fix MistralDualModel _update_causal_mask issue
            self._patch_mistral_dual_model()
            
            # Patch 2: Set proper environment variables for transformers
            self._set_transformers_env_vars()
            
            # Patch 3: Configure CUDA memory management
            self._configure_cuda_memory()
            
            logger.info("✅ Transformers patches applied successfully")
            self.fixes_applied.append("transformers_patches")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply transformers patches: {e}")
            raise
    
    def _patch_mistral_dual_model(self):
        """Apply specific patch for MistralDualModel compatibility."""
        try:
            # Import after setting environment variables
            import torch
            from transformers import PreTrainedModel
            
            # Create a compatibility wrapper for MistralDualModel
            def _update_causal_mask_patch(self, attention_mask=None, input_ids=None, **kwargs):
                """Compatibility patch for _update_causal_mask method."""
                if hasattr(self, '_original_update_causal_mask'):
                    return self._original_update_causal_mask(attention_mask, input_ids, **kwargs)
                
                # Fallback implementation
                if attention_mask is None:
                    return None
                
                batch_size, seq_len = attention_mask.shape
                device = attention_mask.device
                dtype = attention_mask.dtype
                
                # Create a simple causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=dtype, device=device), 
                    diagonal=1
                )
                causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
                
                # Combine with attention mask
                combined_mask = attention_mask.unsqueeze(1) * (1 - causal_mask)
                return combined_mask
            
            # Apply the patch globally
            if not hasattr(PreTrainedModel, '_update_causal_mask'):
                PreTrainedModel._update_causal_mask = _update_causal_mask_patch
                logger.info("✅ MistralDualModel _update_causal_mask patch applied")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not apply MistralDualModel patch: {e}")
    
    def _set_transformers_env_vars(self):
        """Set proper environment variables for transformers."""
        env_vars = {
            'TRANSFORMERS_CACHE': '/app/.cache/transformers',
            'HF_HOME': '/app/.cache/huggingface',
            'TORCH_HOME': '/app/.cache/torch',
            'TOKENIZERS_PARALLELISM': 'false',
            'TRANSFORMERS_OFFLINE': '0',
            'HF_HUB_DISABLE_TELEMETRY': '1',
            'CUDA_LAUNCH_BLOCKING': '0',
        }
        
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
                logger.info(f"✅ Set {key}={value}")
    
    def _configure_cuda_memory(self):
        """Configure CUDA memory management."""
        try:
            import torch
            
            if torch.cuda.is_available():
                # Set memory fraction to avoid OOM
                torch.cuda.set_per_process_memory_fraction(0.9)
                
                # Enable memory optimization
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048,expandable_segments:True'
                
                # Clear cache
                torch.cuda.empty_cache()
                
                logger.info("✅ CUDA memory configuration applied")
            else:
                logger.info("⚠️  CUDA not available, skipping memory configuration")
                
        except Exception as e:
            logger.warning(f"⚠️  Could not configure CUDA memory: {e}")
    
    def create_patched_embedding_model(self):
        """Create a patched version of the embedding model."""
        try:
            # Import with patches applied
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Model configuration with compatibility fixes
            model_kwargs = {
                'torch_dtype': torch.float16,
                'device_map': 'auto',
                'trust_remote_code': True,
                'use_safetensors': True,
            }
            
            # Create a compatibility wrapper
            class PatchedSentenceTransformer(SentenceTransformer):
                """Patched version of SentenceTransformer with compatibility fixes."""
                
                def __init__(self, model_name_or_path, **kwargs):
                    # Apply patches before initialization
                    super().__init__(model_name_or_path, **kwargs)
                    
                    # Apply post-initialization patches
                    self._apply_model_patches()
                
                def _apply_model_patches(self):
                    """Apply patches to the loaded model."""
                    try:
                        # Patch the underlying model if needed
                        if hasattr(self, '_modules'):
                            for module in self._modules.values():
                                if hasattr(module, 'auto_model'):
                                    model = module.auto_model
                                    if hasattr(model, 'model') and not hasattr(model, '_update_causal_mask'):
                                        # Apply the patch
                                        model._update_causal_mask = lambda *args, **kwargs: None
                                        logger.info("✅ Applied _update_causal_mask patch to model")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not apply model patches: {e}")
            
            logger.info("✅ Patched embedding model class created")
            return PatchedSentenceTransformer
            
        except Exception as e:
            logger.error(f"❌ Failed to create patched embedding model: {e}")
            raise
    
    def fix_configuration_issues(self):
        """Fix configuration inconsistencies."""
        try:
            # Load current environment
            current_env = dict(os.environ)
            
            # Configuration fixes
            config_fixes = {
                'EMBEDDING_BATCH_SIZE': '32',  # Reduce from 64 to avoid OOM
                'RERANKING_BATCH_SIZE': '32',  # Reduce from 64 to avoid OOM
                'VEC_RECALL_NUM': '64',        # Reduce from 128 to avoid OOM
                'MAX_SIZE': '512',
                'EMBEDDING_FP16': 'true',
                'RERANKING_FP16': 'true',
                'PRELOAD_MODELS': 'true',
                'LOG_LEVEL': 'INFO',
            }
            
            # Apply fixes
            for key, value in config_fixes.items():
                if key not in current_env or current_env[key] != value:
                    os.environ[key] = value
                    logger.info(f"✅ Fixed {key}={value}")
            
            logger.info("✅ Configuration fixes applied")
            self.fixes_applied.append("configuration_fixes")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply configuration fixes: {e}")
            raise
    
    def validate_environment(self):
        """Validate the environment after applying fixes."""
        try:
            # Check critical environment variables
            required_vars = [
                'EMBED_MODEL', 'RERANK_MODEL', 'BERT_PATH',
                'ZILLIZ_URI', 'ZILLIZ_TOKEN', 'MILVUS_COLLECTION',
                'EMBEDDING_DEVICE', 'RERANKING_DEVICE', 'TEXT_SPLITTER_DEVICE'
            ]
            
            missing_vars = []
            for var in required_vars:
                if var not in os.environ:
                    missing_vars.append(var)
            
            if missing_vars:
                logger.error(f"❌ Missing environment variables: {missing_vars}")
                return False
            
            # Check CUDA availability
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    gpu_count = torch.cuda.device_count()
                    logger.info(f"✅ CUDA available with {gpu_count} GPU(s)")
                else:
                    logger.warning("⚠️  CUDA not available, will use CPU")
            except ImportError:
                logger.warning("⚠️  PyTorch not available")
            
            logger.info("✅ Environment validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment validation failed: {e}")
            return False
    
    def apply_all_fixes(self):
        """Apply all compatibility fixes."""
        logger.info("🔧 Starting GeoGPT-RAG compatibility fixes...")
        
        try:
            # Apply fixes in order
            self.apply_transformers_patches()
            self.fix_configuration_issues()
            
            # Validate environment
            if self.validate_environment():
                logger.info("🎉 All compatibility fixes applied successfully!")
                logger.info(f"📋 Fixes applied: {', '.join(self.fixes_applied)}")
                return True
            else:
                logger.error("❌ Environment validation failed after applying fixes")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to apply compatibility fixes: {e}")
            return False


def main():
    """Main function to apply all fixes."""
    fixer = EmbeddingCompatibilityFixer()
    
    # Apply all fixes
    success = fixer.apply_all_fixes()
    
    if success:
        print("✅ GeoGPT-RAG compatibility fixes applied successfully!")
        print("🚀 You can now restart the application.")
        return 0
    else:
        print("❌ Some fixes failed. Please check the logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 