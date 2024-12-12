from typing import List, Optional, Union
from datetime import datetime
import numpy as np
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import Memory, MemoryStream
from ..utils import load_all_prompts
import os
import logging

logger = logging.getLogger(__name__)

class ImportanceCalculator:
    def __init__(self, model_manager):
        self.time_weight = 0.3
        self.relevance_weight = 0.7
        self.importance_prompt = self._load_importance_prompt()['agent.importance']
        self.model_manager = model_manager
    def _load_importance_prompt(self) -> str:
        try:
            return load_all_prompts()
        except Exception as e:
            logger.error(f"Error loading importance prompt: {e}")
            return ''

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

    async def calculate_memory_importance(self, memory: Memory, model: str) -> float:
        try:
            prompt = self.importance_prompt.format(
                memory_content=memory.content
            )
            response = await self.model_manager.chat_completion(
                model=model,
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

            try:
                result = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return 0.05

            if not isinstance(result, dict):
                logger.error("Response is not a dictionary")
                return 0.05

            importance_score = result.get("importance_score")
            if importance_score is None:
                logger.error("Missing importance_score in response")
                return 0.05

            try:
                importance_score = float(importance_score)
            except (ValueError, TypeError):
                logger.error("Invalid importance_score format")
                return 0.05

            if "reasoning" in result:
                logger.debug(f"Memory importance calculation: {result['reasoning']}")
            return max(0.0, min(1.0, importance_score))
            
        except Exception as e:
            logger.error(f"Error calculating memory importance: {str(e)}")
            return 0.05

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
        
        return similarities if similarities else [0.05] * len(related_memories)