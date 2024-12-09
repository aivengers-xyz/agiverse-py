from typing import List, Optional
import numpy as np
import litellm
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

load_dotenv()

class EmbeddingGenerator:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.dimension = 1536  
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_embedding(self, text: str) -> List[float]:
        try:
            response = await litellm.aembedding(
                model=self.model,
                input=text,
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.dimension
    