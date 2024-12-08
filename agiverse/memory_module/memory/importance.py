from typing import List, Optional, Union
from datetime import datetime
import numpy as np
import litellm
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import Memory, MemoryStream
import os
import logging

logger = logging.getLogger(__name__)

class ImportanceCalculator:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key
        self.model = model or os.getenv('IMPORTANCE_MODEL', 'gpt-4o-mini')
        self.time_weight = 0.3
        self.relevance_weight = 0.7
        
    async def calculate_relevance(self, memory: Memory, current_time: datetime,
                                related_memories: List[Memory]) -> float:
        memory_times = [m.created_at for m in related_memories]
        time_factor = self._calculate_time_decay(memory.created_at, memory_times)
        relevance_factor = await self._calculate_relevance(memory, related_memories)
        
        importance = [
            min(max(self.time_weight * t + self.relevance_weight * r, 0.0), 1.0)
            for t, r in zip(time_factor, relevance_factor)
        ]
        return importance

    async def calculate_memory_importance(self, memory: Memory) -> float:
        current_memory = f"Current memory: {memory.content}"
        prompt = f"""As an expert in evaluating memory importance, analyze the following memory and its context.
        Determine how important this memory is on a scale from 0.0 to 1.0, where:
        - 1.0: Extremely important (critical information, major events, key decisions)
        - 0.7-0.9: Very important (significant events, valuable insights)
        - 0.4-0.6: Moderately important (useful but not critical information)
        - 0.1-0.3: Slightly important (minor details, routine events)
        - 0.0: Not important (trivial or redundant information)

        Consider factors like:
        - Uniqueness of information
        - Potential future relevance
        - Relationship to other memories
        - Impact on decision-making
        - Emotional significance

        {current_memory}

        Return only a JSON object with the format:
        {{"importance_score": float, "reasoning": "brief explanation"}}
        """

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": "You are an expert at evaluating the importance of memories and information."
                },
                {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            importance_score = float(result["importance_score"])
            
            logger.debug(f"Memory importance calculation: {result['reasoning']}")
            
            return max(0.0, min(1.0, importance_score))
            
        except Exception as e:
            logger.error(f"Error calculating memory importance: {e}")
            return 0.3

    def _calculate_time_decay(self, memory_created_at: datetime, 
                         reference_times: Union[datetime, List[datetime]]) -> List[float]:
        if isinstance(reference_times, datetime):
            time_diff = (reference_times - memory_created_at).total_seconds()
            return [1.0 / (1.0 + time_diff / (24 * 3600))]
        
        time_decays = []
        for ref_time in reference_times:
            time_diff = (ref_time - memory_created_at).total_seconds()
            decay = 1.0 / (1.0 + time_diff / (24 * 3600))
            time_decays.append(decay)
            
        return time_decays

    async def _calculate_relevance(self, memory: Memory, related_memories: List[Memory]) -> List[float]:
        if not related_memories or memory.embedding is None:
            return [0.1] 
        similarities = []
        for related_memory in related_memories:
            if related_memory.embedding is not None:
                similarity = np.dot(memory.embedding, related_memory.embedding)
                similarities.append(similarity)
        
        return similarities if similarities else [0.1] * len(related_memories)
