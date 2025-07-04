"""
AWS SageMaker LLM Client

Provides interface for invoking SageMaker endpoints for text generation.
Supports various SageMaker-hosted model formats and handles JSON serialization.
"""

import json
import logging
from typing import Dict, Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

logger = logging.getLogger(__name__)


class SageMakerLLMClient:
    """
    SageMaker LLM client for text generation.
    
    Supports common SageMaker model formats including:
    - Hugging Face Text Generation Inference (TGI)
    - Custom model formats
    - JSON-based request/response handling
    """
    
    def __init__(
        self,
        endpoint_name: str,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        """
        Initialize SageMaker LLM client.
        
        Args:
            endpoint_name: SageMaker endpoint name
            region_name: AWS region
            aws_access_key_id: AWS access key (optional, can use IAM roles)
            aws_secret_access_key: AWS secret key (optional, can use IAM roles)
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for SageMaker integration. "
                "Install with: pip install boto3"
            )
        
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        
        # Initialize SageMaker runtime client
        session_kwargs = {"region_name": region_name}
        
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            })
        
        try:
            self.client = boto3.client("sagemaker-runtime", **session_kwargs)
            logger.info(f"Initialized SageMaker client for endpoint: {endpoint_name}")
        except NoCredentialsError:
            raise RuntimeError(
                "AWS credentials not found. Please configure AWS credentials "
                "via environment variables, IAM roles, or AWS credentials file."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SageMaker client: {e}")
    
    def _prepare_request_payload(
        self,
        prompt: str,
        max_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.8,
        **kwargs
    ) -> str:
        """
        Prepare request payload for SageMaker endpoint.
        
        Supports multiple formats based on model type.
        """
        # Common payload format (works with most HuggingFace TGI endpoints)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "return_full_text": False,
                **kwargs
            }
        }
        
        return json.dumps(payload)
    
    def _parse_response(self, response_body: str) -> str:
        """
        Parse SageMaker response and extract generated text.
        
        Handles multiple response formats:
        - HuggingFace TGI format: [{"generated_text": "..."}]
        - Custom formats: {"text": "...", "output": "...", etc.}
        """
        try:
            response_json = json.loads(response_body)
            
            # Handle HuggingFace TGI format (list of dicts)
            if isinstance(response_json, list) and len(response_json) > 0:
                if "generated_text" in response_json[0]:
                    return response_json[0]["generated_text"]
            
            # Handle dictionary formats
            if isinstance(response_json, dict):
                # Try common field names
                for field in ["generated_text", "text", "output", "response", "result"]:
                    if field in response_json:
                        text = response_json[field]
                        # Handle nested structures
                        if isinstance(text, list) and len(text) > 0:
                            if isinstance(text[0], dict) and "generated_text" in text[0]:
                                return text[0]["generated_text"]
                            return str(text[0])
                        return str(text)
            
            # Fallback: return the whole response as string
            logger.warning("Unexpected response format, returning raw response")
            return str(response_json)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return f"Error: Failed to parse response - {response_body}"
        except Exception as e:
            logger.error(f"Error parsing SageMaker response: {e}")
            return f"Error: Failed to process response - {str(e)}"
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.8,
        timeout: int = 300,
        **kwargs
    ) -> str:
        """
        Generate text using SageMaker endpoint.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            timeout: Request timeout in seconds
            **kwargs: Additional model parameters
            
        Returns:
            Generated text response
        """
        try:
            # Prepare request payload
            payload = self._prepare_request_payload(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            # Invoke SageMaker endpoint
            logger.debug(f"Invoking SageMaker endpoint: {self.endpoint_name}")
            
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=payload,
                # Add timeout if supported by the client
            )
            
            # Read and parse response
            response_body = response["Body"].read().decode("utf-8")
            generated_text = self._parse_response(response_body)
            
            logger.debug(f"SageMaker generation successful, response length: {len(generated_text)}")
            return generated_text
            
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(f"SageMaker ClientError [{error_code}]: {error_message}")
            
            if error_code == "ValidationException":
                return f"Error: Invalid request to SageMaker endpoint - {error_message}"
            elif error_code == "ModelError":
                return f"Error: Model error in SageMaker endpoint - {error_message}"
            elif error_code == "ServiceUnavailable":
                return f"Error: SageMaker service temporarily unavailable - {error_message}"
            else:
                return f"Error: SageMaker error [{error_code}] - {error_message}"
                
        except Exception as e:
            logger.error(f"Unexpected error in SageMaker generation: {e}")
            return f"Error: Failed to generate response - {str(e)}"
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test SageMaker endpoint connectivity and basic functionality.
        
        Returns:
            Dictionary with test results and endpoint information
        """
        try:
            # Try to describe the endpoint
            describe_response = boto3.client(
                "sagemaker", 
                region_name=self.region_name
            ).describe_endpoint(EndpointName=self.endpoint_name)
            
            endpoint_status = describe_response["EndpointStatus"]
            instance_type = describe_response["ProductionVariants"][0]["InstanceType"]
            
            # Test with a simple prompt
            test_prompt = "Hello, this is a test message."
            test_response = self.generate(
                prompt=test_prompt,
                max_tokens=50,
                temperature=0.1
            )
            
            return {
                "status": "success",
                "endpoint_name": self.endpoint_name,
                "endpoint_status": endpoint_status,
                "instance_type": instance_type,
                "test_prompt": test_prompt,
                "test_response": test_response[:200] + "..." if len(test_response) > 200 else test_response,
                "region": self.region_name
            }
            
        except Exception as e:
            return {
                "status": "error",
                "endpoint_name": self.endpoint_name,
                "error": str(e),
                "region": self.region_name
            } 