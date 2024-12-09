from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent import Agent

import asyncio
import json
import logging
import time
import os
import websockets
from datetime import datetime
from .data import DataTypes
from .utils import (
    distance_obj,
    is_valid_state_data,
    is_valid_map_data,
    is_valid_player_data,
    remove_additional_data,
    seconds_since,
    format_json,
    format_memory,
)
from .memory.base import Memory

logger = logging.getLogger(__name__)

class MessagingHandler:
    agent: "Agent"

    def __init__(self, agent):
        self.agent = agent
        self.message_queue = asyncio.Queue()
        self.map_data = None
        self.nearby_map = None
        self.players_data = None
        self.nearby_players = None
        self.assets_data = None
        self.inventory_data = None
        self.state_data = None
        self.available_actions = None
        self.system_messages = []
        self.other_data = []
        self.last_prompt_time = None
        self.use_model = False
        self.min_model_interval = int(os.getenv('MIN_MODEL_INTERVAL', 5))
        self.max_model_interval = int(os.getenv('MAX_MODEL_INTERVAL', 60))
        self.model_interval = self.min_model_interval
        self.action_id = int(time.time() * 1000)

    async def connect(self, uri):
        self.websocket = await websockets.connect(uri)
        return self

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb):
        await self.websocket.close()

    async def handle_messages(self, websocket, stop_event):
        receive_task = asyncio.create_task(self._receive_messages(websocket, stop_event))
        process_task = asyncio.create_task(self._process_messages(websocket, stop_event))

        done, pending = await asyncio.wait(
            [receive_task, process_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            if task.exception():
                raise task.exception()

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Task {task} has been cancelled successfully.")

    async def _receive_messages(self, websocket, stop_event):
        try:
            async for message in websocket:
                if stop_event.is_set():
                    logger.info("Stop event detected. Exiting handle_messages.")
                    break
                await self.message_queue.put(json.loads(message))
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Receive messages - Connection closed: {e}")
            raise

    async def _process_messages(self, websocket, stop_event):
        try:
            while True:
                if stop_event.is_set():
                    logger.info("Stop event detected. Exiting handle_messages.")
                    break

                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=60)
                    if stop_event.is_set():
                        logger.info("Stop event detected. Exiting handle_messages.")
                        break
                    if not await self._process_message(message):
                        continue
                except asyncio.TimeoutError:
                    self.use_model = True
                    message = None
                except asyncio.CancelledError:
                    raise

                if not self.message_queue.empty():
                    continue

                if len(self.system_messages) == 0:
                    seconds_since_last_prompt = seconds_since(self.last_prompt_time) if self.last_prompt_time else self.max_model_interval
                    if seconds_since_last_prompt < self.model_interval:
                        continue
                    if seconds_since_last_prompt < self.max_model_interval and not self.use_model:
                        continue

                try:
                    response = await self._generate_model_response(websocket)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error generating model response: {e}")
                    continue

                try:
                    await self._process_model_response(response)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error processing model response: {e}")
                    continue

                try:
                    await self._send_action(response, websocket)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error sending action: {e}")
                    raise
        except asyncio.CancelledError:
            logger.info("Process messages has been cancelled.")
            raise
        except Exception as e:
            logger.error(f"Process messages encountered an unexpected error: {e}")
            raise

    async def _process_message(self, message):
        msg_type = message.get('type')
        data = message.get('data')

        if msg_type != 'tickEnd':
            logger.info(f"Received: {msg_type} {data}")

        if msg_type == 'map':
            self.map_data = data
            self.use_model = True
        elif msg_type == 'players':
            self.players_data = data
        elif msg_type == 'assets':
            self.assets_data = data
            self.use_model = True
        elif msg_type == 'inventory':
            self.inventory_data = data
            self.use_model = True
        elif msg_type == 'state':
            self.state_data = data
            if self.state_data['state'] in ['moving', 'learning', 'working']:
                self.model_interval = self.max_model_interval if self.state_data['state'] != 'moving' else self.model_interval
            else:
                self.model_interval = self.min_model_interval
            self.use_model = True
        elif msg_type == 'availableActions':
            self.available_actions = data
            self.use_model = True
        elif msg_type == 'tickEnd':
            pass
        elif msg_type == 'system':
            self.system_messages.append(message)
            self.use_model = True
            self.agent._save_data(DataTypes.SYSTEM_MESSAGE, message)
        else:
            self.other_data.append(message)
            self.use_model = True
            self.agent._save_data(DataTypes.SERVER_MESSAGE, message)

        if is_valid_state_data(self.state_data) and is_valid_map_data(self.map_data, 'buildings'):
            nearby_map = {'buildings': [], 'my_houses': []}
            for building in self.map_data['buildings']:
                building['distance'] = distance_obj(building['entrance'], self.state_data)

            sorted_buildings = sorted(
                self.map_data['buildings'],
                key=lambda b: b.get('distance', float('inf'))
            )

            for building in sorted_buildings:
                remove_additional_data(building)
                distance = building.get('distance', float('inf'))
                if distance > self.agent.vision_range_buildings and len(nearby_map['buildings']) >= self.agent.min_num_buildings:
                    break
                nearby_map['buildings'].append(building)

            if is_valid_map_data(self.assets_data, 'rentedBuildings'):
                for building in self.assets_data['rentedBuildings']:
                    building['distance'] = distance_obj(building['entrance'], self.state_data)

                sorted_rented_buildings = sorted(
                    self.assets_data['rentedBuildings'],
                    key=lambda b: b.get('distance', float('inf'))
                )

                for building in sorted_rented_buildings:
                    remove_additional_data(building)
                    distance = building.get('distance', float('inf'))
                    if distance > self.agent.vision_range_buildings and len(nearby_map['my_houses']) >= self.agent.min_num_buildings:
                        break
                    nearby_map['my_houses'].append(building)

            self.nearby_map = nearby_map

        if is_valid_state_data(self.state_data) and is_valid_player_data(self.players_data):
            nearby_players = []
            for player in self.players_data:
                player['distance'] = distance_obj(player, self.state_data)

            sorted_players = sorted(
                self.players_data,
                key=lambda p: p.get('distance', float('inf'))
            )

            for player in sorted_players:
                if player.get('playerID') == self.state_data.get('playerID'):
                    continue
                distance = player.get('distance', float('inf'))
                if distance > self.agent.vision_range_players and len(nearby_players) >= self.agent.min_num_players:
                    break
                nearby_players.append(player)

            self.nearby_players = nearby_players

        return msg_type == 'tickEnd'

    async def _generate_model_response(self, websocket):
        last_memory_step = self.agent.working_memory.get_last_step()
        working_memory_content = last_memory_step.to_string() if last_memory_step else ""
        all_memories = await self.agent.memory_manager.memory_stream.get_all_memories()
        
        long_term_memory = await self._filter_and_rank_memories(working_memory_content, all_memories)
        
        return await self.agent.get_model_response(
            'agent.agent',
            character_name=self.agent.name,
            character_info=self.agent.get_prompt('character.info'),
            nearby_map=format_json(self.nearby_map),
            nearby_players=format_json(self.nearby_players),
            assets=format_json(self.assets_data),
            inventory=format_json(self.inventory_data),
            state=format_json(self.state_data),
            available_actions=format_json(self.available_actions),
            working_memory=self.agent.working_memory.steps_to_string(),
            long_term_memory=long_term_memory,
            planning=self.agent.planning,
            system_messages=format_json(self.system_messages),
            messages=format_json(self.other_data),
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )

    async def _process_model_response(self, response):
        self.agent._save_data(DataTypes.MODEL_RESPONSE, response)

        logger.info('-' * 100)
        logger.info(f"Observation: {response['observation']}")
        logger.info(f"Thought: {response['thought']}")
        logger.info(f"Long term memory: {response.get('longTermMemory', self.agent.long_term_memory)}")
        logger.info(f"Planning: {response.get('planning', self.agent.planning)}")
        logger.info(f"Action: {response['action']}")
        logger.info(f"System message reply action: {response.get('systemMessageReplyAction')}")
        logger.info('-' * 100)

        new_long_term_memory = response.get('longTermMemory', '')
        if new_long_term_memory:
            self.agent.long_term_memory = new_long_term_memory

        new_planning = response.get('planning', '')
        if new_planning:
            self.agent.planning = new_planning
        await self.agent._handle_model_response(response, self.state_data)
        logger.error(f"Working memory size: {len(self.agent.working_memory)}")
        await self.agent.working_memory._compress_steps()
            
        self.system_messages = []
        self.other_data = []
        self.last_prompt_time = datetime.now()
        self.use_model = False
        self.model_interval = self.min_model_interval

    async def _send_action(self, response, websocket):
        if not response.get('action') or response['action'].get('action') == 'no action':
            logger.info('No action to perform.')
        else:
            response['action']['actionID'] = self.action_id
            self.action_id += 1
            logger.info(f"Sending action: {response['action']}")
            await websocket.send(json.dumps(response['action']))

        if response.get('systemMessageReplyAction'):
            logger.info(f"Sending system message reply action: {response['systemMessageReplyAction']}")
            await websocket.send(json.dumps(response['systemMessageReplyAction']))

    async def _filter_and_rank_memories(self, working_memory_content: str, all_memories: List[Memory]) -> List[Memory]:
        if not (working_memory_content and all_memories and self.state_data):
            return []

        current_memory = Memory(
            content=working_memory_content,
            type="working_memory",
            created_at=datetime.now()
        )
        current_memory_embedding = await self.agent.embedding_generator.get_embedding(current_memory.content)
        current_memory.embedding = current_memory_embedding
        
        memory_importance = await self.agent.importance_calculator.calculate_relevance(
            current_memory,
            datetime.now(),
            all_memories
        )
        current_pos = (self.state_data.get('locationX', 0), self.state_data.get('locationY', 0))
        filtered_indices = []
        for i, memory in enumerate(all_memories):
            if not memory.metadata or 'location' not in memory.metadata:
                continue
            memory_pos = (
                memory.metadata['location'].get('locationX', 0),
                memory.metadata['location'].get('locationY', 0)
            )
            
            distance = abs(current_pos[0] - memory_pos[0]) + abs(current_pos[1] - memory_pos[1])
            
            if distance <= self.agent.vision_range_buildings:
                filtered_indices.append(i)
        if filtered_indices:
            filtered_importance = [memory_importance[i] for i in filtered_indices]
            top_k = min(3, len(filtered_indices))
            top_indices = sorted(range(len(filtered_importance)), 
                               key=lambda i: filtered_importance[i],
                               reverse=True)[:top_k]
            return [all_memories[filtered_indices[i]] for i in top_indices]
        else:
            top_k = min(3, len(all_memories))
            top_indices = sorted(range(len(memory_importance)),
                               key=lambda i: memory_importance[i],
                               reverse=True)[:top_k]
            return [all_memories[i] for i in top_indices]

