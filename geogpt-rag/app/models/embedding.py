import torch
import os
import warnings
from typing import List, Optional

# Suppress warnings that clutter the output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

DEFAULT_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class EmbeddingModel:
    """Wrapper around Sentence‑Transformers that mimics the GEOembedding API.

    The public methods mirror the original /query and /passage endpoints so we can
    keep exactly the same prompting logic while running *in‑process* instead of
    over HTTP.
    
    Includes compatibility fixes for MistralDualModel and newer transformers versions.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        task_description: str = DEFAULT_TASK,
        fp16: bool = False,
    ) -> None:
        # Apply compatibility fixes before loading model
        self._apply_compatibility_fixes()
        
        # Lazy import to avoid issues during module loading
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(f"sentence_transformers is required but not installed: {e}")
        
        # Configure model kwargs with compatibility settings
        model_kwargs = {}
        if fp16:
            model_kwargs["torch_dtype"] = torch.float16
        
        # Add trust_remote_code for custom models
        model_kwargs["trust_remote_code"] = True
        
        # Add device_map for better memory management
        if device == "cuda" and torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        
        try:
            # Initialize model with error handling
            self.model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs=model_kwargs or None,
            )
            
            # Apply post-initialization patches
            self._patch_loaded_model()
            
        except Exception as e:
            print(f"⚠️  Error loading model {model_name}: {e}")
            print("🔧 Attempting to load with fallback configuration...")
            
            # Fallback configuration
            fallback_kwargs = {"trust_remote_code": True}
            self.model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs=fallback_kwargs,
            )
            self._patch_loaded_model()
            
        self.task = task_description

    def _apply_compatibility_fixes(self):
        """Apply compatibility fixes for transformers and other dependencies."""
        try:
            # Set environment variables for compatibility
            env_fixes = {
                'TOKENIZERS_PARALLELISM': 'false',
                'TRANSFORMERS_OFFLINE': '0',
                'HF_HUB_DISABLE_TELEMETRY': '1',
            }
            
            for key, value in env_fixes.items():
                if key not in os.environ:
                    os.environ[key] = value
            
            # Apply the MistralDualModel patch
            self._patch_mistral_dual_model()
            
        except Exception as e:
            print(f"⚠️  Could not apply all compatibility fixes: {e}")
    
    def _patch_mistral_dual_model(self):
        """Apply specific patch for MistralDualModel compatibility."""
        try:
            from transformers import PreTrainedModel
            
            # Create a compatibility wrapper for MistralDualModel
            def _update_causal_mask_patch(self, attention_mask=None, input_ids=None, **kwargs):
                """Compatibility patch for _update_causal_mask method."""
                # For embedding models, we don't need causal masking
                # Just return None to avoid the error
                return None
            
            # Apply the patch globally to PreTrainedModel
            if not hasattr(PreTrainedModel, '_update_causal_mask'):
                PreTrainedModel._update_causal_mask = _update_causal_mask_patch
                print("✅ Applied MistralDualModel _update_causal_mask compatibility patch")
            
        except Exception as e:
            print(f"⚠️  Could not apply MistralDualModel patch: {e}")
    
    def _patch_loaded_model(self):
        """Apply patches to the loaded model."""
        try:
            # Patch the underlying model if needed
            if hasattr(self.model, '_modules'):
                for module in self.model._modules.values():
                    if hasattr(module, 'auto_model'):
                        model = module.auto_model
                        if hasattr(model, 'model') and not hasattr(model, '_update_causal_mask'):
                            # Apply the patch directly to the model
                            model._update_causal_mask = lambda *args, **kwargs: None
                            print("✅ Applied _update_causal_mask patch to loaded model")
                            
        except Exception as e:
            print(f"⚠️  Could not apply model patches: {e}")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _wrap_instruction(task: str, text: str) -> str:
        """Replicates the original get_detailed_instruct() helper."""
        return f"Instruct: {task}\nQuery: {text}"

    def _encode(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Encode texts with error handling and memory management."""
        try:
            processed = (
                [self._wrap_instruction(instruction or self.task, t) for t in texts]
                if instruction or self.task
                else texts
            )
            
            # Encode with proper error handling
            embeddings = self.model.encode(
                processed,
                convert_to_tensor=False,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,  # Reduce output noise
            )
            
            return embeddings.tolist()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️  GPU out of memory, reducing batch size from {batch_size} to {batch_size//2}")
                torch.cuda.empty_cache()
                return self._encode(texts, instruction, batch_size//2)
            else:
                raise e
        except Exception as e:
            print(f"❌ Encoding error: {e}")
            raise ConnectionError(f"Embedding model error: {e}")

    # ------------------------------------------------------------------
    # Public API (mirrors HTTP micro‑service)
    # ------------------------------------------------------------------
    def encode_queries(
        self,
        queries: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Encode queries with proper error handling."""
        return self._encode(queries, instruction, batch_size)

    def encode_passages(
        self,
        passages: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Encode passages with proper error handling."""
        # Passages only get wrapped if instruction is explicitly provided
        if instruction:
            processed = [self._wrap_instruction(instruction, p) for p in passages]
        else:
            processed = passages  # No wrapping for passages without instruction
        
        try:
            embeddings = self.model.encode(
                processed,
                convert_to_tensor=False,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,  # Reduce output noise
            )
            
            return embeddings.tolist()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️  GPU out of memory, reducing batch size from {batch_size} to {batch_size//2}")
                torch.cuda.empty_cache()
                return self.encode_passages(passages, instruction, batch_size//2)
            else:
                raise e
        except Exception as e:
            print(f"❌ Passage encoding error: {e}")
            raise ConnectionError(f"Embedding model error: {e}")

    # Backwards‑compat: old code calls .encode() generically
    def encode(self, texts, **kwargs):  # type: ignore[override]
        """Encode texts generically (backwards compatibility)."""
        return self.encode_queries(texts, **kwargs)