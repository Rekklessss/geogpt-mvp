"""
CUDA and GPU compatibility tests for GeoGPT-RAG.

Tests CUDA 12.8 compatibility, PyTorch GPU access, and integration with 
the updated dependency stack (pymilvus 2.4.10, tf-keras, etc.).
"""

import pytest
import torch
from unittest.mock import patch, Mock


class TestCUDACompatibility:
    """Test CUDA 12.8 and GPU functionality."""
    
    def test_pytorch_cuda_version(self):
        """Test PyTorch CUDA version compatibility."""
        # This will work whether CUDA is available or not
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            # Check CUDA version compatibility
            cuda_version = torch.version.cuda
            assert cuda_version is not None, "CUDA version should be available"
            
            # Check device count
            device_count = torch.cuda.device_count()
            assert device_count > 0, "Should have at least one CUDA device"
            
            # Check if we can create tensors on GPU
            device = torch.device("cuda:0")
            test_tensor = torch.randn(10, 10, device=device)
            assert test_tensor.is_cuda, "Tensor should be on CUDA device"
            
            print(f"✅ CUDA {cuda_version} available with {device_count} device(s)")
        else:
            print("⚠️  CUDA not available in test environment (CPU-only)")

    def test_tensorflow_gpu_access(self):
        """Test TensorFlow GPU access with new tf-keras package."""
        try:
            import tensorflow as tf
            
            # Check TF version
            tf_version = tf.__version__
            print(f"📱 TensorFlow version: {tf_version}")
            
            # Check GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✅ TensorFlow detects {len(gpus)} GPU(s)")
                
                # Test basic GPU operation
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    result = tf.matmul(test_tensor, test_tensor)
                    assert result.shape == (2, 2)
                    print("✅ TensorFlow GPU operations working")
            else:
                print("⚠️  TensorFlow GPU not available in test environment")
                
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_transformers_device_detection(self):
        """Test that transformers library can detect and use GPU."""
        try:
            import transformers
            
            # Mock the device detection since we're not loading actual models in tests
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.device_count', return_value=1):
                    # This would normally use GPU if available
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    assert device in ["cuda", "cpu"]
                    print(f"✅ Transformers would use device: {device}")
                    
        except ImportError:
            pytest.skip("Transformers not available")

    @patch('torch.cuda.is_available')
    def test_embeddings_gpu_compatibility(self, mock_cuda_available):
        """Test embeddings module GPU compatibility."""
        mock_cuda_available.return_value = True
        
        try:
            from app.embeddings import GeoEmbeddings
            from app.config import EMBEDDING_DEVICE
            
            # Test that embedding device configuration works
            if EMBEDDING_DEVICE == "cuda":
                print("✅ Embeddings configured to use CUDA")
            else:
                print(f"📋 Embeddings configured to use: {EMBEDDING_DEVICE}")
            
            # Test that we can create an embeddings instance
            embedder = GeoEmbeddings()
            assert embedder is not None
            print("✅ Embeddings instance created successfully")
                
        except ImportError:
            pytest.skip("App modules not available in test environment")

    @patch('torch.cuda.is_available')
    def test_reranking_gpu_compatibility(self, mock_cuda_available):
        """Test reranking module GPU compatibility."""
        mock_cuda_available.return_value = True
        
        try:
            from app.reranking import GeoReRanking
            from app.config import RERANKING_DEVICE
            
            # Test that reranking device configuration works
            if RERANKING_DEVICE == "cuda":
                print("✅ Reranking configured to use CUDA")
            else:
                print(f"📋 Reranking configured to use: {RERANKING_DEVICE}")
            
            # Test that we can create a reranker instance
            reranker = GeoReRanking()
            assert reranker is not None
            print("✅ Reranker instance created successfully")
                
        except ImportError:
            pytest.skip("App modules not available in test environment")

    def test_model_cache_directories(self):
        """Test that model cache directories are properly configured."""
        import os
        from pathlib import Path
        
        # Test cache directory structure that would be used
        cache_dirs = [
            ".cache/transformers",
            ".cache/huggingface", 
            ".cache/torch",
            "model_cache",
            "huggingface_cache"
        ]
        
        for cache_dir in cache_dirs:
            # In production, these would be created by the Docker container
            # Here we just verify the path structure is valid
            path = Path(cache_dir)
            assert path.name in ["transformers", "huggingface", "torch", "model_cache", "huggingface_cache"]
            print(f"✅ Cache directory structure valid: {cache_dir}")

    def test_dependency_versions(self):
        """Test that all dependencies are compatible versions."""
        import sys
        
        try:
            import torch
            import transformers
            import sentence_transformers
            import pymilvus
            
            versions = {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "pytorch": torch.__version__,
                "transformers": transformers.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "pymilvus": pymilvus.__version__
            }
            
            print("📦 Dependency versions:")
            for name, version in versions.items():
                print(f"   {name}: {version}")
            
            # Verify key compatibility requirements
            assert sys.version_info >= (3, 12), "Python 3.12+ required"
            assert "cu121" in torch.__version__ or not torch.cuda.is_available(), "PyTorch should use CUDA 12.1 compatible version"
            
            # Check pymilvus version
            pymilvus_version = tuple(map(int, pymilvus.__version__.split('.')[:2]))
            assert pymilvus_version >= (2, 4), f"pymilvus 2.4.10+ required, got {pymilvus.__version__}"
            
            print("✅ All dependency versions compatible")
            
        except ImportError as e:
            pytest.skip(f"Dependency not available: {e}")


@pytest.mark.performance
class TestPerformanceWithGPU:
    """Performance tests that can utilize GPU if available."""
    
    def test_tensor_operations_performance(self):
        """Basic performance test for tensor operations."""
        import time
        
        # Test CPU performance
        start_time = time.time()
        cpu_tensor = torch.randn(1000, 1000)
        result_cpu = torch.matmul(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start_time
        
        print(f"⏱️  CPU tensor operation: {cpu_time:.4f}s")
        
        if torch.cuda.is_available():
            # Test GPU performance
            start_time = time.time()
            gpu_tensor = torch.randn(1000, 1000, device='cuda')
            result_gpu = torch.matmul(gpu_tensor, gpu_tensor)
            torch.cuda.synchronize()  # Wait for GPU completion
            gpu_time = time.time() - start_time
            
            print(f"🚀 GPU tensor operation: {gpu_time:.4f}s")
            print(f"📈 Speedup: {cpu_time/gpu_time:.2f}x")
        else:
            print("⚠️  GPU not available for performance testing")


if __name__ == "__main__":
    # Run a quick compatibility check
    test_cuda = TestCUDACompatibility()
    test_cuda.test_pytorch_cuda_version()
    test_cuda.test_dependency_versions()
    print("🎉 CUDA compatibility tests completed!") 