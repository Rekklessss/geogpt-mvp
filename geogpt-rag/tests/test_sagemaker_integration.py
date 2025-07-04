"""
Comprehensive test suite for AWS SageMaker LLM integration.

Tests SageMaker endpoint connectivity, response parsing, error handling,
and integration with the RAG pipeline.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from app.models.sagemaker_llm import SageMakerLLMClient
from app.kb import generate_llm_response, _generate_sagemaker_response
from app.config import SAGEMAKER_ENDPOINT_NAME, AWS_REGION


class TestSageMakerLLMClient:
    """Test SageMaker LLM client functionality."""
    
    @patch('app.models.sagemaker_llm.boto3.client')
    def test_client_initialization_success(self, mock_boto_client):
        """Test successful SageMaker client initialization."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        client = SageMakerLLMClient(
            endpoint_name="test-endpoint",
            region_name="us-east-1"
        )
        
        assert client.endpoint_name == "test-endpoint"
        assert client.region_name == "us-east-1"
        assert client.client == mock_client
        
        # Verify boto3 client was called with correct parameters
        mock_boto_client.assert_called_once_with(
            "sagemaker-runtime",
            region_name="us-east-1"
        )

    @patch('app.models.sagemaker_llm.boto3.client')
    def test_client_initialization_with_credentials(self, mock_boto_client):
        """Test SageMaker client initialization with explicit credentials."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        client = SageMakerLLMClient(
            endpoint_name="test-endpoint",
            region_name="us-east-1",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret"
        )
        
        # Verify boto3 client was called with credentials
        mock_boto_client.assert_called_once_with(
            "sagemaker-runtime",
            region_name="us-east-1",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret"
        )

    @patch('app.models.sagemaker_llm.BOTO3_AVAILABLE', False)
    def test_client_initialization_boto3_missing(self):
        """Test error when boto3 is not available."""
        with pytest.raises(ImportError, match="boto3 is required"):
            SageMakerLLMClient("test-endpoint")

    @patch('app.models.sagemaker_llm.boto3.client')
    def test_client_initialization_no_credentials(self, mock_boto_client):
        """Test error when AWS credentials are not available."""
        mock_boto_client.side_effect = NoCredentialsError()
        
        with pytest.raises(RuntimeError, match="AWS credentials not found"):
            SageMakerLLMClient("test-endpoint")

    def test_prepare_request_payload(self):
        """Test request payload preparation for different formats."""
        with patch('app.models.sagemaker_llm.boto3.client'):
            client = SageMakerLLMClient("test-endpoint")
            
            payload = client._prepare_request_payload(
                prompt="Test prompt",
                max_tokens=1000,
                temperature=0.8,
                top_p=0.9
            )
            
            payload_dict = json.loads(payload)
            assert payload_dict["inputs"] == "Test prompt"
            assert payload_dict["parameters"]["max_new_tokens"] == 1000
            assert payload_dict["parameters"]["temperature"] == 0.8
            assert payload_dict["parameters"]["top_p"] == 0.9
            assert payload_dict["parameters"]["do_sample"] is True
            assert payload_dict["parameters"]["return_full_text"] is False

    def test_parse_response_huggingface_format(self):
        """Test parsing HuggingFace TGI response format."""
        with patch('app.models.sagemaker_llm.boto3.client'):
            client = SageMakerLLMClient("test-endpoint")
            
            # HuggingFace TGI format
            response_body = json.dumps([{"generated_text": "This is the generated response"}])
            result = client._parse_response(response_body)
            
            assert result == "This is the generated response"

    def test_parse_response_dict_format(self):
        """Test parsing dictionary response formats."""
        with patch('app.models.sagemaker_llm.boto3.client'):
            client = SageMakerLLMClient("test-endpoint")
            
            # Dictionary with 'text' field
            response_body = json.dumps({"text": "Generated text response"})
            result = client._parse_response(response_body)
            assert result == "Generated text response"
            
            # Dictionary with 'output' field
            response_body = json.dumps({"output": "Generated output response"})
            result = client._parse_response(response_body)
            assert result == "Generated output response"

    def test_parse_response_malformed_json(self):
        """Test parsing malformed JSON response."""
        with patch('app.models.sagemaker_llm.boto3.client'):
            client = SageMakerLLMClient("test-endpoint")
            
            result = client._parse_response("invalid json")
            assert "Error: Failed to parse response" in result

    @patch('app.models.sagemaker_llm.boto3.client')
    def test_generate_success(self, mock_boto_client):
        """Test successful text generation."""
        # Setup mock client
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock successful endpoint response
        mock_response = {
            "Body": Mock()
        }
        mock_response["Body"].read.return_value = json.dumps([{
            "generated_text": "This is a test response from SageMaker"
        }]).encode("utf-8")
        
        mock_client.invoke_endpoint.return_value = mock_response
        
        # Test generation
        client = SageMakerLLMClient("test-endpoint")
        result = client.generate(
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.7
        )
        
        assert result == "This is a test response from SageMaker"
        
        # Verify endpoint was called correctly
        mock_client.invoke_endpoint.assert_called_once_with(
            EndpointName="test-endpoint",
            ContentType="application/json",
            Body=json.dumps({
                "inputs": "Test prompt",
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "do_sample": True,
                    "return_full_text": False
                }
            })
        )

    @patch('app.models.sagemaker_llm.boto3.client')
    def test_generate_client_error(self, mock_boto_client):
        """Test handling of SageMaker client errors."""
        # Setup mock client with error
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "Invalid endpoint name"
            }
        }
        mock_client.invoke_endpoint.side_effect = ClientError(error_response, "InvokeEndpoint")
        
        # Test error handling
        client = SageMakerLLMClient("test-endpoint")
        result = client.generate("Test prompt")
        
        assert "SageMaker API error" in result
        assert "ValidationException" in result
        assert "Invalid endpoint name" in result

    @patch('app.models.sagemaker_llm.boto3.client')
    def test_generate_general_error(self, mock_boto_client):
        """Test handling of general errors during generation."""
        # Setup mock client with general error
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.invoke_endpoint.side_effect = Exception("Network error")
        
        # Test error handling
        client = SageMakerLLMClient("test-endpoint")
        result = client.generate("Test prompt")
        
        assert "Error generating text" in result
        assert "Network error" in result

    @patch('app.models.sagemaker_llm.boto3.client')
    def test_test_connection_success(self, mock_boto_client):
        """Test successful connection test."""
        # Setup mock client
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock successful test response
        mock_response = {
            "Body": Mock()
        }
        mock_response["Body"].read.return_value = json.dumps([{
            "generated_text": "Connection test successful"
        }]).encode("utf-8")
        
        mock_client.invoke_endpoint.return_value = mock_response
        
        # Test connection
        client = SageMakerLLMClient("test-endpoint")
        result = client.test_connection()
        
        assert result["status"] == "success"
        assert result["endpoint"] == "test-endpoint"
        assert "response_time" in result


class TestSageMakerIntegration:
    """Test SageMaker integration with the main application."""
    
    @patch('app.kb.SAGEMAKER_ENDPOINT_NAME', 'test-sagemaker-endpoint')
    @patch('app.kb.LLM_PROVIDER', 'sagemaker')
    @patch('app.models.sagemaker_llm.boto3.client')
    def test_generate_sagemaker_response_success(self, mock_boto_client):
        """Test successful SageMaker response generation."""
        # Setup mock client
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock successful response
        mock_response = {
            "Body": Mock()
        }
        mock_response["Body"].read.return_value = json.dumps([{
            "generated_text": "This is a SageMaker generated response about geological processes."
        }]).encode("utf-8")
        
        mock_client.invoke_endpoint.return_value = mock_response
        
        # Test response generation
        result = _generate_sagemaker_response("What are the main geological processes?")
        
        assert result == "This is a SageMaker generated response about geological processes."

    @patch('app.kb.SAGEMAKER_ENDPOINT_NAME', '')
    @patch('app.kb.LLM_PROVIDER', 'sagemaker')
    def test_generate_sagemaker_response_no_endpoint(self):
        """Test SageMaker response when endpoint is not configured."""
        result = _generate_sagemaker_response("Test prompt")
        
        assert "SageMaker endpoint is not properly configured" in result

    @patch('app.kb.LLM_PROVIDER', 'sagemaker')
    def test_generate_llm_response_sagemaker_provider(self):
        """Test LLM response generation routes to SageMaker."""
        with patch('app.kb._generate_sagemaker_response') as mock_sagemaker:
            mock_sagemaker.return_value = "SageMaker response"
            
            result = generate_llm_response("Test prompt")
            
            assert result == "SageMaker response"
            mock_sagemaker.assert_called_once_with("Test prompt")

    @patch('app.kb.LLM_PROVIDER', 'openai-compatible')
    def test_generate_llm_response_openai_provider(self):
        """Test LLM response generation routes to OpenAI-compatible."""
        with patch('app.kb._generate_openai_compatible_response') as mock_openai:
            mock_openai.return_value = "OpenAI response"
            
            result = generate_llm_response("Test prompt")
            
            assert result == "OpenAI response"
            mock_openai.assert_called_once_with("Test prompt")


class TestSageMakerRAGIntegration:
    """Test SageMaker integration within the full RAG pipeline."""
    
    @patch('app.kb.LLM_PROVIDER', 'sagemaker')
    @patch('app.models.sagemaker_llm.boto3.client')
    def test_full_rag_query_with_sagemaker(self, mock_boto_client):
        """Test complete RAG query using SageMaker LLM."""
        # Setup mock SageMaker client
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock SageMaker response
        mock_response = {
            "Body": Mock()
        }
        mock_response["Body"].read.return_value = json.dumps([{
            "generated_text": "Based on the retrieved documents [citation:1][citation:2], geological processes include volcanic activity and erosion."
        }]).encode("utf-8")
        
        mock_client.invoke_endpoint.return_value = mock_response
        
        # Mock KB instance with SageMaker integration
        with patch('app.kb.KBDocQA') as MockKB:
            mock_kb = MockKB.return_value
            mock_kb.retrieval.return_value = [
                {"text": "Volcanic activity shapes terrain", "score": 0.9},
                {"text": "Erosion wears down mountains", "score": 0.8}
            ]
            
            # Mock query method to use real generate_llm_response
            def mock_query(query, k=3, expand_len=1024, score_threshold=1.5):
                docs = mock_kb.retrieval.return_value
                if docs:
                    docs_text = "\n".join([f"[document {i} begin]{doc['text']}[document {i} end]" 
                                         for i, doc in enumerate(docs)])
                    prompt = f"Search results: {docs_text}\nQuestion: {query}"
                    response = generate_llm_response(prompt)
                    return docs, response
                else:
                    return [], generate_llm_response(query)
            
            mock_kb.query.side_effect = mock_query
            
            # Test full RAG query
            kb = MockKB()
            docs, response = kb.query("What are the main geological processes?")
            
            # Verify SageMaker was used and response includes citations
            assert len(docs) == 2
            assert "geological processes" in response
            assert "[citation:" in response
            mock_client.invoke_endpoint.assert_called_once()


@pytest.mark.asyncio
class TestSageMakerAPIIntegration:
    """Test SageMaker integration through the FastAPI endpoints."""
    
    @patch('app.main.kb_instance')
    @patch('app.kb.LLM_PROVIDER', 'sagemaker')
    @patch('app.models.sagemaker_llm.boto3.client')
    async def test_api_query_with_sagemaker(self, mock_boto_client, mock_kb):
        """Test API query endpoint using SageMaker LLM."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Setup mock SageMaker client
        mock_sagemaker_client = Mock()
        mock_boto_client.return_value = mock_sagemaker_client
        
        # Mock SageMaker response
        mock_response = {
            "Body": Mock()
        }
        mock_response["Body"].read.return_value = json.dumps([{
            "generated_text": "This is a SageMaker-powered RAG response about climate patterns."
        }]).encode("utf-8")
        
        mock_sagemaker_client.invoke_endpoint.return_value = mock_response
        
        # Mock KB query method
        mock_docs = [{"text": "Climate data", "score": 0.9}]
        mock_kb.query.return_value = (mock_docs, "SageMaker response")
        
        # Test API query
        response = client.post("/query", json={
            "query": "What are the climate patterns?",
            "k": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "response" in data
        assert "documents" in data

    @patch('app.main.kb_instance')
    @patch('app.kb.LLM_PROVIDER', 'sagemaker')
    @patch('app.kb.SAGEMAKER_ENDPOINT_NAME', '')
    async def test_api_query_sagemaker_not_configured(self, mock_kb):
        """Test API graceful handling when SageMaker is not configured."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Mock KB query to return fallback response
        mock_kb.query.return_value = ([], "SageMaker endpoint is not properly configured")
        
        # Test API query
        response = client.post("/query", json={
            "query": "Test query",
            "k": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "not properly configured" in data["response"]


# Utility functions for SageMaker testing
def create_mock_sagemaker_response(text: str) -> Dict[str, Any]:
    """Create a mock SageMaker response for testing."""
    return {
        "Body": Mock(read=Mock(return_value=json.dumps([{
            "generated_text": text
        }]).encode("utf-8")))
    }


def create_mock_sagemaker_error(error_code: str, message: str) -> ClientError:
    """Create a mock SageMaker ClientError for testing."""
    error_response = {
        "Error": {
            "Code": error_code,
            "Message": message
        }
    }
    return ClientError(error_response, "InvokeEndpoint") 