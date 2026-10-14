# Created by Marinara and Claude Sonnet 4
# Shoutout to Il Dottore, my beloved.

# Imports
import datetime
import speech_recognition as sr
import io
import tempfile
import discord
from discord import app_commands
import anthropic
import asyncio
import json
import os
from dotenv import load_dotenv
from typing import Dict, List, Set, Tuple, Optional
import aiohttp
import random
import re
import base64
from abc import ABC, abstractmethod
from google import genai
from google.genai import types # type: ignore
import openai
from openai import AsyncOpenAI
from collections import defaultdict
import time
import logging
import warnings

# Suppress warnings BEFORE importing pydub
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", category=RuntimeWarning)

from pydub import AudioSegment

# Suppress Discord connection warnings/errors
logging.getLogger('discord').setLevel(logging.CRITICAL)
logging.getLogger('aiohttp').setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Only show critical errors
logging.basicConfig(level=logging.CRITICAL)

# Environment setup
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CUSTOM_API_KEY = os.getenv('CUSTOM_API_KEY')
# OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

if not DISCORD_TOKEN:
    print("Error: DISCORD_TOKEN environment variable not set.")
    exit(1)

# Discord client setup
intents = discord.Intents.default()
intents.message_content = True
intents.emojis = True
intents.members = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# Data persistence paths
DATA_DIR = "bot_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Remove/replace existing prompt-related file paths and variables

DM_TOGGLE_FILE = os.path.join(DATA_DIR, "dm_toggle.json")
DM_LAST_INTERACTION_FILE = os.path.join(DATA_DIR, "dm_last_interaction.json")
DM_LORE_FILE = os.path.join(DATA_DIR, "dm_lore.json")
DM_MEMORIES_FILE = os.path.join(DATA_DIR, "dm_memories.json")
PERSONALITIES_FILE = os.path.join(DATA_DIR, "personalities.json")
HISTORY_LENGTHS_FILE = os.path.join(DATA_DIR, "history_lengths.json")
LORE_FILE = os.path.join(DATA_DIR, "lore.json")
ACTIVITY_FILE = os.path.join(DATA_DIR, "activity.json")
AUTONOMOUS_FILE = os.path.join(DATA_DIR, "autonomous.json")
MEMORIES_FILE = os.path.join(DATA_DIR, "memories.json")
TEMPERATURE_FILE = os.path.join(DATA_DIR, "temperature.json")
WELCOME_SENT_FILE = os.path.join(DATA_DIR, "welcome_sent.json")
DM_SERVER_SELECTION_FILE = os.path.join(DATA_DIR, "dm_server_selection.json")
DM_ENABLED_FILE = os.path.join(DATA_DIR, "dm_enabled.json")
VISION_CACHE_FILE = os.path.join(DATA_DIR, "vision_cache.json")

# New format settings files
FORMAT_SETTINGS_FILE = os.path.join(DATA_DIR, "format_settings.json")
DM_FORMAT_SETTINGS_FILE = os.path.join(DATA_DIR, "dm_format_settings.json")
SERVER_FORMAT_DEFAULTS_FILE = os.path.join(DATA_DIR, "server_format_defaults.json")
NSFW_SETTINGS_FILE = os.path.join(DATA_DIR, "nsfw_settings.json")
DM_NSFW_SETTINGS_FILE = os.path.join(DATA_DIR, "dm_nsfw_settings.json")
CUSTOM_FORMAT_INSTRUCTIONS_FILE = os.path.join(DATA_DIR, "custom_format_instructions.json")
PREFILL_SETTINGS_FILE = os.path.join(DATA_DIR, "prefill_settings.json")

# Files for old prompt system - TO BE REMOVED
CUSTOM_PROMPTS_FILE = os.path.join(DATA_DIR, "custom_prompts.json")
# Removed: PROMPT_SETTINGS_FILE and DM_PROMPT_SETTINGS_FILE - no longer needed

# AI Provider Classes
class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict], system_prompt: str, temperature: float = 1.0, model: str = None, max_tokens: int = 2000) -> str:
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class ClaudeProvider(AIProvider):
    """Claude AI provider using Anthropic API"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
    
    async def generate_response(self, messages: List[Dict], system_prompt: str, temperature: float = 1.0, model: str = None, max_tokens: int = 2000) -> str:
        if not self.api_key:
            return "❌ Claude API key not configured. Please contact the bot administrator."
        
        try:
            model = model or self.get_default_model()
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
                stream=False
            )
            
            response_text = response.content[0].text
            
            # Check if the response contains proxy or API errors
            if any(error_indicator in response_text.lower() for error_indicator in [
                "proxy error", "upstream connect error", "connection termination", 
                "service unavailable", "context size limit", "request validation failed",
                "tokens.*exceeds", "http 503", "http 400", "http 429", "rate limit", "timeout"
            ]):
                return f"❌ Claude API error: {response_text}"
            
            # Clean any base64 data from the response
            response_text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}', '[IMAGE DATA REMOVED]', response_text)
            response_text = re.sub(r'\b[A-Za-z0-9+/=]{100,}\b', '[BASE64 DATA REMOVED]', response_text)
            
            return response_text
        except Exception as e:
            return f"❌ Claude API error: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        return [
            "claude-opus-4-1",   # Vision support
            "claude-opus-4",     # Vision support
            "claude-opus-4-0",
            "claude-sonnet-4-0", 
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest"
        ]
    
    def get_default_model(self) -> str:
        return "claude-3-7-sonnet-latest"
    
    def is_available(self) -> bool:
        return bool(self.api_key)

class GeminiProvider(AIProvider):
    """Gemini AI provider using Google's API"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        if api_key:
            self.client = genai.Client(api_key=api_key)
    
    async def generate_response(self, messages: List[Dict], system_prompt: str, temperature: float = 1.0, model: str = None, max_output_tokens: int = 2000) -> str:
        if not self.api_key:
            return "❌ Gemini API key not configured. Please contact the bot administrator."
        
        try:
            model = model or self.get_default_model()
            
            # Convert messages to Gemini format
            gemini_messages = []
            for i, msg in enumerate(messages):
                try:
                    role = "user" if msg["role"] == "user" else "model"
                    content = msg["content"]
                    
                    if isinstance(content, list):
                        # Complex content with text and images
                        parts = []
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    parts.append({"text": part["text"]})
                                elif part.get("type") == "image":
                                    # Convert image to Gemini format
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": part["media_type"],
                                            "data": part["data"]
                                        }
                                    })
                        
                        if parts:
                            gemini_messages.append({"role": role, "parts": parts})
                    
                    elif isinstance(content, str) and content.strip():
                        # Simple text content
                        gemini_messages.append({"role": role, "parts": [{"text": content}]})
                        
                except Exception as msg_error:
                    print(f"Gemini: Error processing message {i}: {msg_error}")
                    continue
            
            # Ensure we have at least one message and it ends with user
            if not gemini_messages:
                gemini_messages = [{"role": "user", "parts": [{"text": "Hello"}]}]
            elif gemini_messages[-1]["role"] != "user":
                gemini_messages.append({"role": "user", "parts": [{"text": "Continue the conversation naturally."}]})
            
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_prompt,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF"),
                ],
            )
            
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=gemini_messages,
                config=generation_config
            )
            
            if hasattr(response, 'text') and response.text:
                # Check if the response contains proxy or API errors
                if any(error_indicator in response.text.lower() for error_indicator in [
                    "proxy error", "upstream connect error", "connection termination", 
                    "service unavailable", "context size limit", "request validation failed",
                    "tokens.*exceeds", "http 503", "http 400", "http 429", "rate limit", "timeout"
                ]):
                    return f"❌ Gemini API error: {response.text}"
                
                # Clean any base64 data from the response
                clean_text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}', '[IMAGE DATA REMOVED]', response.text)
                clean_text = re.sub(r'\b[A-Za-z0-9+/=]{100,}\b', '[BASE64 DATA REMOVED]', clean_text)
                return clean_text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                text_parts.append(part.text)
                        if text_parts:
                            response_text = "".join(text_parts)
                            # Check if the response contains proxy or API errors
                            if any(error_indicator in response_text.lower() for error_indicator in [
                                "proxy error", "upstream connect error", "connection termination", 
                                "service unavailable", "context size limit", "request validation failed",
                                "tokens.*exceeds", "http 503", "http 400", "http 429", "rate limit", "timeout"
                            ]):
                                return f"❌ Gemini API error: {response_text}"
                            
                            # Clean any base64 data from the response
                            response_text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}', '[IMAGE DATA REMOVED]', response_text)
                            response_text = re.sub(r'\b[A-Za-z0-9+/=]{100,}\b', '[BASE64 DATA REMOVED]', response_text)
                            return response_text
                            
                if hasattr(candidate, 'finish_reason'):
                    if candidate.finish_reason == "SAFETY":
                        return "❌ Gemini response blocked by safety filters. Try rephrasing your request."
                    elif candidate.finish_reason == "MAX_TOKENS":
                        return "❌ Gemini response was cut off due to token limit."
                    elif candidate.finish_reason == "RECITATION":
                        return "❌ Gemini response blocked due to recitation concerns."
                    else:
                        return f"❌ Gemini stopped generation: {candidate.finish_reason}"
            
            return "❌ Gemini returned empty response (no text generated)"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"❌ Gemini API error: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        return [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]
    
    def get_default_model(self) -> str:
        return "gemini-2.5-flash"
    
    def is_available(self) -> bool:
        return bool(self.api_key)

class OpenAIProvider(AIProvider):
    """OpenAI provider supporting ChatGPT models"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        if api_key:
            self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate_response(self, messages: List[Dict], system_prompt: str, temperature: float = 1.0, model: str = None, max_tokens: int = 2000) -> str:
        if not self.api_key:
            return "❌ OpenAI API key not configured. Please contact the bot administrator."
        
        try:
            model = model or self.get_default_model()
            
            # Convert messages to OpenAI format
            formatted_messages = [{"role": "system", "content": system_prompt}]
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if isinstance(content, list):
                    # Complex content with text and images
                    openai_content = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                openai_content.append({
                                    "type": "text",
                                    "text": part["text"]
                                })
                            elif part.get("type") == "image_url":
                                # Already in OpenAI format
                                openai_content.append(part)
                            elif part.get("type") == "image":
                                # Convert from other formats (shouldn't happen, but just in case)
                                if "data" in part and "media_type" in part:
                                    openai_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{part['media_type']};base64,{part['data']}",
                                            "detail": "high"
                                        }
                                    })
                    
                    if openai_content:
                        formatted_messages.append({"role": role, "content": openai_content})
                
                elif isinstance(content, str) and content.strip():
                    # Simple text content
                    formatted_messages.append({"role": role, "content": content})
            
            # Check if model supports vision
            vision_models = ["gpt-5", "gpt-5-chat-latest", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview", "gpt-4.1", "gpt-4.1-mini"]
            supports_vision = any(vision_model in model.lower() for vision_model in vision_models)
            
            # If model doesn't support vision but we have images, convert them to text descriptions
            if not supports_vision:
                for message in formatted_messages:
                    if isinstance(message.get("content"), list):
                        text_parts = []
                        for part in message["content"]:
                            if part.get("type") == "text":
                                text_parts.append(part["text"])
                            elif part.get("type") == "image_url":
                                text_parts.append("[Image was provided but this model doesn't support vision]")
                        message["content"] = " ".join(text_parts)
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            response_text = response.choices[0].message.content
            
            # Check if the response contains proxy or API errors
            if any(error_indicator in response_text.lower() for error_indicator in [
                "proxy error", "upstream connect error", "connection termination", 
                "service unavailable", "context size limit", "request validation failed",
                "tokens.*exceeds", "http 503", "http 400", "http 429", "rate limit", "timeout"
            ]):
                return f"❌ OpenAI API error: {response_text}"
            
            # Clean any base64 data from the response
            response_text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}', '[IMAGE DATA REMOVED]', response_text)
            response_text = re.sub(r'\b[A-Za-z0-9+/=]{100,}\b', '[BASE64 DATA REMOVED]', response_text)
            
            return response_text
            
        except Exception as e:
            return f"❌ OpenAI API error: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        return [
            "gpt-5",             # Vision support
            "gpt-5-chat-latest", # Vision support
            "gpt-4.1",           # Vision support
            "gpt-4.1-mini",      # Vision support  
            "gpt-4.1-nano",      # Vision support
            "o3-preview",        # Vision support
            "gpt-4o",            # Vision support
            "gpt-4o-mini",       # Vision support
            "gpt-4-turbo",       # Vision support
            "gpt-4",             # Vision support
            "gpt-3.5-turbo",     # No vision
            "o1-preview",        # No vision
            "o1-mini"            # No vision
        ]
    
    def get_default_model(self) -> str:
        return "gpt-4.1"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def supports_vision(self, model: str) -> bool:
        """Check if a model supports vision/images"""
        vision_models = [
            "gpt-5", "gpt-5-chat-latest", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", 
            "gpt-4-vision-preview", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", 
            "o3-preview", "gpt-4"
        ]
        return any(vision_model in model.lower() for vision_model in vision_models)

class CustomProvider(AIProvider):
    """Custom provider for local/self-hosted models"""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:1234/v1"):
        super().__init__(api_key)
        self.base_url = base_url
        self._vision_cache = self.load_vision_cache()  # Load from persistent storage
        if api_key:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url
            )
    
    def load_vision_cache(self) -> Dict[str, bool]:
        """Load vision support cache from file"""
        try:
            if os.path.exists(VISION_CACHE_FILE):
                with open(VISION_CACHE_FILE, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def save_vision_cache(self):
        """Save vision support cache to file with error handling"""
        try:
            # Ensure the directory exists
            os.makedirs(DATA_DIR, exist_ok=True)
            
            # Always save, even if cache is empty
            with open(VISION_CACHE_FILE, 'w') as f:
                json.dump(self._vision_cache, f, indent=2)
            
            # print(f"Vision cache saved successfully with {len(self._vision_cache)} entries")
            
        except Exception as e:
            print(f"Error saving vision cache: {e}")
            # Don't raise the error, just log it

    def load_vision_cache(self) -> Dict[str, bool]:
        """Load vision support cache from file with error handling"""
        try:
            if os.path.exists(VISION_CACHE_FILE):
                with open(VISION_CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    # print(f"Vision cache loaded successfully with {len(cache_data)} entries")
                    return cache_data
            else:
                # print("No existing vision cache file found, starting fresh")
                # Create empty file to ensure it exists
                self._vision_cache = {}
                self.save_vision_cache()
                return {}
        except Exception as e:
            # print(f"Error loading vision cache: {e}")
            return {}

    async def supports_vision_dynamic(self, model: str) -> bool:
        """Dynamically check if a model supports vision by testing with a small image"""
        
        # Check if client is available
        if not self.api_key or not hasattr(self, 'client'):
            return False
        
        # Check persistent cache first
        cache_key = f"{self.base_url}:{model}"
        if cache_key in self._vision_cache:
            return self._vision_cache[cache_key]
        
        # print(f"Testing vision support for {model} (first time)...")
        
        try:
            # Create a minimal test message with a tiny image
            test_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Can you see this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            # 1x1 white pixel PNG as base64 (super tiny)
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                            "detail": "low"
                        }
                    }
                ]
            }]
            
            # Try to make a request with minimal tokens to save costs
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=test_messages,
                    max_tokens=5,  # Very minimal response
                    temperature=0  # Deterministic
                )
                
                # If we get here without error, the model supports vision
                # print(f"✅ {model} supports vision!")
                self._vision_cache[cache_key] = True
                self.save_vision_cache()
                return True
                
            except Exception as e:
                error_str = str(e).lower()
                # print(f"Testing {model} vision support failed: {error_str}")
                
                # Check for specific vision-related errors
                vision_error_indicators = [
                    "vision", "image", "multimodal", "unsupported content type",
                    "invalid content", "image_url not supported", "images are not supported",
                    "does not support images", "visual", "multimedia", "unsupported message type",
                    "content type not supported", "image content is not supported"
                ]
                
                if any(keyword in error_str for keyword in vision_error_indicators):
                    # This suggests the model exists but doesn't support vision
                    # print(f"❌ {model} doesn't support vision (confirmed)")
                    self._vision_cache[cache_key] = False
                    self.save_vision_cache()
                    return False
                else:
                    # For OpenRouter Gemini, let's assume it supports vision if the error isn't vision-specific
                    if "gemini" in model.lower() and self.base_url and "openrouter" in self.base_url.lower():
                        # print(f"✅ Assuming {model} on OpenRouter supports vision")
                        self._vision_cache[cache_key] = True
                        self.save_vision_cache()
                        return True
                    
                    # Other errors might be temporary, don't cache
                    # print(f"⚠️ {model} vision test inconclusive (error: {error_str})")
                    return False
                    
        except Exception as outer_error:
            # print(f"Unexpected error during vision testing for {model}: {outer_error}")
            # For OpenRouter Gemini, assume vision support if we can't test
            if "gemini" in model.lower() and self.base_url and "openrouter" in self.base_url.lower():
                # print(f"✅ Assuming {model} on OpenRouter supports vision (test failed)")
                self._vision_cache[cache_key] = True
                self.save_vision_cache()
                return True
            return False
    
    async def generate_response(self, messages: List[Dict], system_prompt: str, temperature: float = 1.0, model: str = None, max_tokens: int = 2000) -> str:
        if not self.api_key:
            return "❌ Custom API key not configured. Please contact the bot administrator."
        
        try:
            model = model or self.get_default_model()
            
            formatted_messages = [{"role": "system", "content": system_prompt}]
            has_images = False
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if isinstance(content, list):
                    # Complex content with text and images
                    openai_content = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                openai_content.append({
                                    "type": "text",
                                    "text": part["text"]
                                })
                            elif part.get("type") == "image_url":
                                has_images = True
                                # Keep OpenAI format as-is for custom providers
                                openai_content.append(part)
                            elif part.get("type") == "image":
                                has_images = True
                                # Convert other formats to OpenAI format
                                if "source" in part and "data" in part["source"]:
                                    # Claude format to OpenAI
                                    openai_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{part['source']['media_type']};base64,{part['source']['data']}",
                                            "detail": "high"
                                        }
                                    })
                                elif "data" in part and "media_type" in part:
                                    # Gemini format to OpenAI
                                    openai_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{part['media_type']};base64,{part['data']}",
                                            "detail": "high"
                                        }
                                    })
                    
                    if openai_content:
                        formatted_messages.append({"role": role, "content": openai_content})
                
                elif isinstance(content, str) and content.strip():
                    formatted_messages.append({"role": role, "content": content})
            
            # If we have images, test vision support only once
            if has_images:
                supports_vision = await self.supports_vision_dynamic(model)
                if not supports_vision:
                    # print(f"Converting images to text for {model} (no vision support)")
                    # Convert images to text descriptions
                    for message in formatted_messages:
                        if isinstance(message.get("content"), list):
                            text_parts = []
                            for part in message["content"]:
                                if part.get("type") == "text":
                                    text_parts.append(part["text"])
                                elif part.get("type") == "image_url":
                                    text_parts.append("[Image was provided but this model doesn't support vision!]")
                            message["content"] = " ".join(text_parts)
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            response_text = response.choices[0].message.content
            
            # Check if the response contains proxy or API errors
            if any(error_indicator in response_text.lower() for error_indicator in [
                "proxy error", "upstream connect error", "connection termination", 
                "service unavailable", "context size limit", "request validation failed",
                "tokens.*exceeds", "http 503", "http 400", "http 429", "rate limit", "timeout"
            ]):
                return f"❌ Custom API error: {response_text}"
            
            # Clean any base64 data from the response
            response_text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}', '[IMAGE DATA REMOVED]', response_text)
            response_text = re.sub(r'\b[A-Za-z0-9+/=]{100,}\b', '[BASE64 DATA REMOVED]', response_text)
            
            return response_text
        
        except Exception as e:
            return f"❌ Custom API error: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        return [
            "auto-detect",
            "custom-model",
            "local-model"
        ]
    
    def get_default_model(self) -> str:
        return "auto-detect"
    
    def is_available(self) -> bool:
        return bool(self.api_key)

class AIProviderManager:
    """Manages different AI providers and user selections"""
    
    def __init__(self):
        self.providers = {
            "claude": ClaudeProvider(CLAUDE_API_KEY),
            "gemini": GeminiProvider(GEMINI_API_KEY),
            "openai": OpenAIProvider(OPENAI_API_KEY),
            "custom": CustomProvider(CUSTOM_API_KEY),
        }
        
        # Cache for custom providers with different URLs
        self.custom_provider_cache = {}
        
        self.guild_provider_settings: Dict[int, str] = {}
        self.guild_model_settings: Dict[int, str] = {}
        self.guild_custom_urls: Dict[int, str] = {}
        
        self.load_settings()
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Get dictionary of provider names and their availability status"""
        return {
            provider_name: provider.is_available() 
            for provider_name, provider in self.providers.items()
        }
    
    def get_provider_models(self, provider_name: str) -> List[str]:
        """Get available models for a specific provider"""
        if provider_name in self.providers:
            return self.providers[provider_name].get_available_models()
        return []
    
    def get_guild_settings(self, guild_id: int) -> Tuple[str, str]:
        """Get provider and model settings for a guild"""
        provider = self.guild_provider_settings.get(guild_id, "claude")
        model = self.guild_model_settings.get(guild_id)
        
        # If no model is set, use the provider's default
        if not model and provider in self.providers:
            model = self.providers[provider].get_default_model()
        
        return provider, model
    
    def get_guild_custom_url(self, guild_id: int) -> str:
        """Get custom URL for a guild (for custom provider)"""
        return self.guild_custom_urls.get(guild_id, "http://localhost:1234/v1")
    
    def set_guild_provider(self, guild_id: int, provider: str, model: str = None, custom_url: str = None) -> bool:
        """Set provider and model for a guild"""
        try:
            if provider not in self.providers:
                return False
            
            self.guild_provider_settings[guild_id] = provider
            
            if model:
                self.guild_model_settings[guild_id] = model
            else:
                # Use provider's default model
                self.guild_model_settings[guild_id] = self.providers[provider].get_default_model()
            
            if provider == "custom" and custom_url:
                self.guild_custom_urls[guild_id] = custom_url
            
            self.save_settings()
            return True
        except Exception:
            return False
    
    def get_custom_provider(self, custom_url: str) -> CustomProvider:
        """Get or create a custom provider instance for the given URL"""
        if custom_url not in self.custom_provider_cache:
            # print(f"Creating new CustomProvider instance for URL: {custom_url}")
            self.custom_provider_cache[custom_url] = CustomProvider(CUSTOM_API_KEY, custom_url)
        # else:
            # print(f"Reusing existing CustomProvider instance for URL: {custom_url}")
        
        return self.custom_provider_cache[custom_url]
    
    async def generate_response(self, messages: List[Dict], system_prompt: str, 
                            temperature: float = 1.0, user_id: int = None, 
                            guild_id: int = None, is_dm: bool = False, max_tokens: int = 2000) -> str:
        """Generate response using appropriate provider"""
        
        # For DMs, check if user has selected a specific server, otherwise use shared guild
        if is_dm:
            selected_guild_id = dm_server_selection.get(user_id)
            if selected_guild_id:
                provider_name, model_name = self.get_guild_settings(selected_guild_id)
            elif guild_id:
                # Fall back to shared guild's model
                provider_name, model_name = self.get_guild_settings(guild_id)
            else:
                # No provider available
                return "❌ No AI provider is configured. Please ensure you're in a server with the bot that has a configured AI provider. If you are, use `/dm_server_select` here to set it up."
        elif guild_id:
            provider_name, model_name = self.get_guild_settings(guild_id)
        else:
            # No guild context and no provider
            return "❌ No AI provider is configured. Please contact the bot administrator to set up API keys."
        
        # ========== PROVIDER DEBUG LOGGING ==========
        print(f"\n🔌 PROVIDER MANAGER DEBUG:")
        print(f"   Provider: {provider_name}")
        print(f"   Model: {model_name}")
        print(f"   Max Tokens: {max_tokens}")
        print(f"   Messages to send: {len(messages)}")
        
        # Log the exact payload being sent to provider
        if provider_name == "custom":
            url_guild_id = dm_server_selection.get(user_id) if is_dm and user_id in dm_server_selection else guild_id
            custom_url = self.get_guild_custom_url(url_guild_id) if url_guild_id else "http://localhost:1234/v1"
            print(f"   Custom URL: {custom_url}")
        
        print("   📦 Sending to AI provider...")
        # ========== END PROVIDER DEBUG LOGGING ==========

        # Handle custom provider with cached instances
        if provider_name == "custom":
            # For DMs, use the selected server's custom URL, otherwise use current guild
            url_guild_id = dm_server_selection.get(user_id) if is_dm and user_id in dm_server_selection else guild_id
            if url_guild_id:
                custom_url = self.get_guild_custom_url(url_guild_id)
                # Use cached provider instance instead of creating new one
                custom_provider = self.get_custom_provider(custom_url)
                return await custom_provider.generate_response(messages, system_prompt, temperature, model_name, max_tokens)
            else:
                # Use default custom provider
                provider = self.providers.get(provider_name)
                return await provider.generate_response(messages, system_prompt, temperature, model_name, max_tokens)
        
        # Handle other providers normally
        provider = self.providers.get(provider_name)
        if not provider or not provider.is_available():
            # No fallback - just return error
            return "❌ No AI providers are available. Please contact the bot administrator to configure API keys."
        
        return await provider.generate_response(messages, system_prompt, temperature, model_name, max_tokens)
    
    def save_settings(self):
        """Save provider settings to files"""
        try:
            guild_settings = {
                "providers": {str(k): v for k, v in self.guild_provider_settings.items()},
                "models": {str(k): v for k, v in self.guild_model_settings.items()},
                "custom_urls": {str(k): v for k, v in self.guild_custom_urls.items()}
            }
            with open(os.path.join(DATA_DIR, "guild_ai_settings.json"), 'w') as f:
                json.dump(guild_settings, f, indent=2)
                
        except Exception:
            pass
    
    def load_settings(self):
        """Load provider settings from files"""
        try:
            guild_file = os.path.join(DATA_DIR, "guild_ai_settings.json")
            if os.path.exists(guild_file):
                with open(guild_file, 'r') as f:
                    data = json.load(f)
                self.guild_provider_settings = {int(k): v for k, v in data.get("providers", {}).items()}
                self.guild_model_settings = {int(k): v for k, v in data.get("models", {}).items()}
                self.guild_custom_urls = {int(k): v for k, v in data.get("custom_urls", {}).items()}
        except Exception:
            pass

# Initialize AI Provider Manager
ai_manager = AIProviderManager()

# Remove all existing system prompts - they will be replaced with the new structure

# New format style constants
FORMAT_CONVERSATIONAL = "conversational"
FORMAT_ASTERISK = "asterisk"
FORMAT_NARRATIVE = "narrative"

# Valid format styles
VALID_FORMAT_STYLES = [FORMAT_CONVERSATIONAL, FORMAT_ASTERISK, FORMAT_NARRATIVE]

class MemoryManager:
    """Manages conversation memories for contextual recall"""
    def __init__(self):
        self.memories: Dict[int, List[Dict]] = {}
        self.dm_memories: Dict[int, List[Dict]] = {}
        self.load_data()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for memory indexing"""
        
        # Handle None or empty text
        if not text or not isinstance(text, str):
            return []
        
        # Clean the text and convert to lowercase
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words and filter
        words = cleaned_text.split()
        
        # Remove common stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
            'what', 'where', 'when', 'why', 'how', 'who', 'which', 'said', 'says', 'just', 'like',
            'get', 'got', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew', 'think', 'thought',
            'take', 'took', 'make', 'made', 'give', 'gave', 'tell', 'told', 'ask', 'asked',
            'discord', 'server', 'channel', 'conversation', 'summary', 'between', 'users', 'bots'
        }
        
        # Filter out stop words, short words, and extract meaningful keywords
        keywords = []
        for word in words:
            if (len(word) >= 3 and 
                word not in stop_words and 
                not word.isdigit() and
                word not in keywords):
                keywords.append(word)
        
        # Limit to reasonable number of keywords
        return keywords[:20]
    
    def save_memory(self, guild_id: int, memory_text: str, keywords: List[str] = None):
        """Save a memory for servers"""
        if guild_id not in self.memories:
            self.memories[guild_id] = []
        
        # Handle None or empty memory_text
        if not memory_text or not isinstance(memory_text, str):
            memory_text = "Empty memory"
        
        if not keywords:
            keywords = self._extract_keywords(memory_text)
        
        memory_entry = {
            "memory": memory_text,
            "keywords": [k.lower() for k in keywords],
            "timestamp": int(time.time())
        }
        
        self.memories[guild_id].append(memory_entry)
        self.save_data()
        return len(self.memories[guild_id]) - 1

    def save_dm_memory(self, user_id: int, memory_text: str, keywords: List[str] = None):
        """Save a memory for DMs"""
        if user_id not in self.dm_memories:
            self.dm_memories[user_id] = []
        
        # Handle None or empty memory_text
        if not memory_text or not isinstance(memory_text, str):
            memory_text = "Empty memory"
        
        if not keywords:
            keywords = self._extract_keywords(memory_text)
        
        memory_entry = {
            "memory": memory_text,
            "keywords": [k.lower() for k in keywords],
            "timestamp": int(time.time())
        }
    
        self.dm_memories[user_id].append(memory_entry)
        self.save_dm_data()
        return len(self.dm_memories[user_id]) - 1

    def save_data(self):
        """Persist server memories to file"""
        try:
            json_data = {str(guild_id): guild_memories for guild_id, guild_memories in self.memories.items()}
            with open(MEMORIES_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass

    def save_dm_data(self):
        """Persist DM memories to file"""
        try:
            json_data = {str(user_id): user_memories for user_id, user_memories in self.dm_memories.items()}
            with open(DM_MEMORIES_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass

    def load_data(self):
        """Load memories from files"""
        try:
            # Load server memories
            if os.path.exists(MEMORIES_FILE):
                with open(MEMORIES_FILE, 'r') as f:
                    json_data = json.load(f)
                self.memories = {int(guild_id_str): guild_memories for guild_id_str, guild_memories in json_data.items()}
            
            # Load DM memories
            if os.path.exists(DM_MEMORIES_FILE):
                with open(DM_MEMORIES_FILE, 'r') as f:
                    json_data = json.load(f)
                self.dm_memories = {int(user_id_str): user_memories for user_id_str, user_memories in json_data.items()}
        except Exception:
            self.memories = {}
            self.dm_memories = {}

    def search_memories(self, guild_id: int, query: str) -> List[Dict]:
        """Search for relevant memories based on query keywords (servers)"""
        if guild_id not in self.memories:
            return []
        
        query_words = [word.lower() for word in query.split()]
        relevant_memories = []
        
        for memory in self.memories[guild_id]:
            for query_word in query_words:
                for keyword in memory["keywords"]:
                    if query_word in keyword or keyword in query_word:
                        if memory not in relevant_memories:
                            relevant_memories.append(memory)
                        break
        
        relevant_memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return relevant_memories[:1]

    def search_dm_memories(self, user_id: int, query: str) -> List[Dict]:
        """Search for relevant memories based on query keywords (DMs)"""
        if user_id not in self.dm_memories:
            return []
        
        query_words = [word.lower() for word in query.split()]
        relevant_memories = []
        
        for memory in self.dm_memories[user_id]:
            for query_word in query_words:
                for keyword in memory["keywords"]:
                    if query_word in keyword or keyword in query_word:
                        if memory not in relevant_memories:
                            relevant_memories.append(memory)
                        break
        
        relevant_memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return relevant_memories[:1]

    def get_all_memories(self, guild_id: int) -> List[Dict]:
        """Get all memories for a guild"""
        return self.memories.get(guild_id, [])

    def get_all_dm_memories(self, user_id: int) -> List[Dict]:
        """Get all memories for a DM"""
        return self.dm_memories.get(user_id, [])

    def delete_all_memories(self, guild_id: int):
        """Delete all memories for a guild"""
        if guild_id in self.memories:
            del self.memories[guild_id]
        self.save_data()

    def delete_all_dm_memories(self, user_id: int):
        """Delete all memories for a DM"""
        if user_id in self.dm_memories:
            del self.dm_memories[user_id]
        self.save_dm_data()

    def edit_memory(self, guild_id: int, memory_index: int, new_memory_text: str) -> bool:
        """Edit a specific memory by index (servers)"""
        if guild_id not in self.memories:
            return False
        
        if not (0 <= memory_index < len(self.memories[guild_id])):
            return False
        
        keywords = self._extract_keywords(new_memory_text)
        self.memories[guild_id][memory_index]["memory"] = new_memory_text
        self.memories[guild_id][memory_index]["keywords"] = [k.lower() for k in keywords]
        self.memories[guild_id][memory_index]["timestamp"] = int(asyncio.get_event_loop().time())
        
        self.save_data()
        return True

    def edit_dm_memory(self, user_id: int, memory_index: int, new_memory_text: str) -> bool:
        """Edit a specific memory by index (DMs)"""
        if user_id not in self.dm_memories:
            return False
        
        if not (0 <= memory_index < len(self.dm_memories[user_id])):
            return False
        
        keywords = self._extract_keywords(new_memory_text)
        self.dm_memories[user_id][memory_index]["memory"] = new_memory_text
        self.dm_memories[user_id][memory_index]["keywords"] = [k.lower() for k in keywords]
        self.dm_memories[user_id][memory_index]["timestamp"] = int(asyncio.get_event_loop().time())
        
        self.save_dm_data()
        return True

    def delete_memory(self, guild_id: int, memory_index: int) -> bool:
        """Delete a specific memory by index (servers)"""
        if guild_id not in self.memories:
            return False
        
        if not (0 <= memory_index < len(self.memories[guild_id])):
            return False
        
        del self.memories[guild_id][memory_index]
        self.save_data()
        return True

    def delete_dm_memory(self, user_id: int, memory_index: int) -> bool:
        """Delete a specific memory by index (DMs)"""
        if user_id not in self.dm_memories:
            return False
        
        if not (0 <= memory_index < len(self.dm_memories[user_id])):
            return False
        
        del self.dm_memories[user_id][memory_index]
        self.save_dm_data()
        return True

    def get_memory_by_index(self, guild_id: int, memory_index: int) -> dict:
        """Get a specific memory by index (servers)"""
        if guild_id not in self.memories:
            return None
        
        if not (0 <= memory_index < len(self.memories[guild_id])):
            return None
        
        return self.memories[guild_id][memory_index]

    def get_dm_memory_by_index(self, user_id: int, memory_index: int) -> dict:
        """Get a specific memory by index (DMs)"""
        if user_id not in self.dm_memories:
            return None
        
        if not (0 <= memory_index < len(self.dm_memories[user_id])):
            return None
        
        return self.dm_memories[user_id][memory_index]

    def save_dm_data(self):
        """Persist DM memories to file"""
        try:
            json_data = {str(user_id): user_memories for user_id, user_memories in self.dm_memories.items()}
            with open(DM_MEMORIES_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass

    def load_data(self):
        """Load memories from files"""
        try:
            # Load server memories
            if os.path.exists(MEMORIES_FILE):
                with open(MEMORIES_FILE, 'r') as f:
                    json_data = json.load(f)
                self.memories = {int(guild_id_str): guild_memories for guild_id_str, guild_memories in json_data.items()}
            
            # Load DM memories
            if os.path.exists(DM_MEMORIES_FILE):
                with open(DM_MEMORIES_FILE, 'r') as f:
                    json_data = json.load(f)
                self.dm_memories = {int(user_id_str): user_memories for user_id_str, user_memories in json_data.items()}
        except Exception:
            self.memories = {}
            self.dm_memories = {}

class LoreBook:
    """Manages character lore entries for server members"""
    def __init__(self):
        self.entries: Dict[int, Dict[str, str]] = {}
        self.dm_entries: Dict[int, str] = {}
        self.load_data()

    def add_entry(self, guild_id: int, user_id: int, lore: str):
        """Add or update lore for a user in servers"""
        if guild_id not in self.entries:
            self.entries[guild_id] = {}
        self.entries[guild_id][user_id] = lore
        self.save_data()

    def add_dm_entry(self, user_id: int, lore: str):
        """Add or update DM-specific lore for a user"""
        self.dm_entries[user_id] = lore
        self.save_dm_data()

    def get_entry(self, guild_id: int, user_id: int) -> str:
        """Get lore for a specific user in servers"""
        return self.entries.get(guild_id, {}).get(user_id, "")

    def get_dm_entry(self, user_id: int) -> str:
        """Get DM-specific lore for a user"""
        return self.dm_entries.get(user_id, "")

    def remove_entry(self, guild_id: int, user_id: int):
        """Remove lore for a user in servers"""
        if guild_id in self.entries and user_id in self.entries[guild_id]:
            del self.entries[guild_id][user_id]
            self.save_data()

    def remove_dm_entry(self, user_id: int):
        """Remove DM-specific lore for a user"""
        if user_id in self.dm_entries:
            del self.dm_entries[user_id]
            self.save_dm_data()

    def save_data(self):
        """Persist server lore data to file"""
        try:
            json_data = {}
            for guild_id, guild_entries in self.entries.items():
                json_data[str(guild_id)] = {str(user_id): lore for user_id, lore in guild_entries.items()}
            
            with open(LORE_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass

    def save_dm_data(self):
        """Persist DM lore data to file"""
        try:
            json_data = {str(user_id): lore for user_id, lore in self.dm_entries.items()}
            with open(DM_LORE_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass

    def load_data(self):
        """Load lore data from files"""
        try:
            # Load server lore
            if os.path.exists(LORE_FILE):
                with open(LORE_FILE, 'r') as f:
                    json_data = json.load(f)
                
                self.entries = {}
                for guild_id_str, guild_entries in json_data.items():
                    guild_id = int(guild_id_str)
                    self.entries[guild_id] = {int(user_id_str): lore for user_id_str, lore in guild_entries.items()}
            
            # Load DM lore
            if os.path.exists(DM_LORE_FILE):
                with open(DM_LORE_FILE, 'r') as f:
                    json_data = json.load(f)
                self.dm_entries = {int(user_id_str): lore for user_id_str, lore in json_data.items()}
        except Exception:
            self.entries = {}
            self.dm_entries = {}

class DMManager:
    """Manages DM-specific features"""
    def __init__(self):
        self.dm_toggle_settings: Dict[int, bool] = {}
        self.last_interactions: Dict[int, float] = {}
        self.pending_check_ups: Set[int] = set()
        self.dm_personalities: Dict[int, tuple] = {}
        self.dm_full_history: Dict[int, bool] = {}
        self.check_up_sent: Dict[int, bool] = {}
        self.load_data()
    
    def set_dm_toggle(self, user_id: int, enabled: bool):
        """Enable/disable auto check-up messages for a user"""
        self.dm_toggle_settings[user_id] = enabled
        if enabled:
            self.update_last_interaction(user_id)
        else:
            self.pending_check_ups.discard(user_id)
        self.save_data()
    
    def is_dm_toggle_enabled(self, user_id: int) -> bool:
        """Check if auto check-up is enabled for a user"""
        return self.dm_toggle_settings.get(user_id, False)
    
    def update_last_interaction(self, user_id: int):
        """Update the last interaction timestamp for a user"""
        self.last_interactions[user_id] = asyncio.get_event_loop().time()
        self.pending_check_ups.discard(user_id)
        # Reset check-up sent flag when user becomes active again
        self.check_up_sent[user_id] = False
        self.save_data()
    
    def get_users_needing_check_up(self) -> List[int]:
        """Get list of users who need a check-up message"""
        current_time = asyncio.get_event_loop().time()
        six_hours = 6 * 60 * 60
        
        users_needing_check_up = []
        
        for user_id, enabled in self.dm_toggle_settings.items():
            if (enabled and 
                user_id not in self.pending_check_ups and
                user_id in self.last_interactions and
                not self.check_up_sent.get(user_id, False) and
                current_time - self.last_interactions[user_id] >= six_hours):
                users_needing_check_up.append(user_id)
        
        return users_needing_check_up
    
    def mark_check_up_sent(self, user_id: int):
        """Mark that a check-up message has been sent"""
        self.pending_check_ups.add(user_id)
        self.check_up_sent[user_id] = True
        self.save_data()
    
    def set_dm_full_history(self, user_id: int, enabled: bool):
        """Enable/disable full history loading for DMs"""
        self.dm_full_history[user_id] = enabled
        self.save_data()
    
    def is_dm_full_history_enabled(self, user_id: int) -> bool:
        """Check if full history loading is enabled for user"""
        return self.dm_full_history.get(user_id, False)
    
    def save_data(self):
        """Persist DM manager data to files"""
        try:
            json_data = {str(user_id): enabled for user_id, enabled in self.dm_toggle_settings.items()}
            with open(DM_TOGGLE_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            json_data = {str(user_id): timestamp for user_id, timestamp in self.last_interactions.items()}
            with open(DM_LAST_INTERACTION_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            dm_personalities_file = os.path.join(DATA_DIR, "dm_personalities.json")
            json_data = {str(user_id): {"guild_id": guild_id, "personality": personality} 
                        for user_id, (guild_id, personality) in self.dm_personalities.items()}
            with open(dm_personalities_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            dm_history_file = os.path.join(DATA_DIR, "dm_full_history.json")
            json_data = {str(user_id): enabled for user_id, enabled in self.dm_full_history.items()}
            with open(dm_history_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            check_up_sent_file = os.path.join(DATA_DIR, "check_up_sent.json")
            json_data = {str(user_id): sent for user_id, sent in self.check_up_sent.items()}
            with open(check_up_sent_file, 'w') as f:
                json.dump(json_data, f, indent=2)
                
        except Exception:
            pass
    
    def load_data(self):
        """Load DM manager data from files"""
        try:
            if os.path.exists(DM_TOGGLE_FILE):
                with open(DM_TOGGLE_FILE, 'r') as f:
                    json_data = json.load(f)
                self.dm_toggle_settings = {int(user_id_str): enabled for user_id_str, enabled in json_data.items()}
            
            if os.path.exists(DM_LAST_INTERACTION_FILE):
                with open(DM_LAST_INTERACTION_FILE, 'r') as f:
                    json_data = json.load(f)
                self.last_interactions = {int(user_id_str): float(timestamp) for user_id_str, timestamp in json_data.items()}
            
            dm_personalities_file = os.path.join(DATA_DIR, "dm_personalities.json")
            if os.path.exists(dm_personalities_file):
                with open(dm_personalities_file, 'r') as f:
                    json_data = json.load(f)
                self.dm_personalities = {int(user_id_str): (data["guild_id"], data["personality"]) 
                                       for user_id_str, data in json_data.items()}
            
            dm_history_file = os.path.join(DATA_DIR, "dm_full_history.json")
            if os.path.exists(dm_history_file):
                with open(dm_history_file, 'r') as f:
                    json_data = json.load(f)
                self.dm_full_history = {int(user_id_str): enabled for user_id_str, enabled in json_data.items()}
            
            check_up_sent_file = os.path.join(DATA_DIR, "check_up_sent.json")
            if os.path.exists(check_up_sent_file):
                with open(check_up_sent_file, 'r') as f:
                    json_data = json.load(f)
                self.check_up_sent = {int(user_id_str): sent for user_id_str, sent in json_data.items()}
            else:
                self.check_up_sent = {}
        except Exception:
            self.dm_toggle_settings = {}
            self.last_interactions = {}
            self.dm_personalities = {}
            self.dm_full_history = {}
            self.check_up_sent = {}

class RequestQueue:
    """Manages queued requests with safe-locking to prevent spam responses"""
    
    def __init__(self):
        self.queues: Dict[int, List] = defaultdict(list)
        self.processing: Dict[int, bool] = defaultdict(bool)
        self.locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        
    async def add_request(self, channel_id: int, message: discord.Message, content: str, 
                        guild: discord.Guild, attachments: List[discord.Attachment],
                        user_name: str, is_dm: bool, user_id: int, reply_to_name: str = None) -> bool:
        """Add a request to the queue. Returns True if added, False if duplicate/spam"""
        # print(f"DEBUG: add_request called for channel {channel_id}, content={repr(content)}")
        
        async with self.locks[channel_id]:
            # Check for recent duplicate requests from same user (spam prevention)
            current_time = time.time()
            for queued_request in self.queues[channel_id]:
                if (queued_request['user_id'] == user_id and 
                    current_time - queued_request['timestamp'] < 3 and
                    queued_request['content'].strip() == content.strip()):  # Check content similarity
                    return False
            
            # Check if user has too many pending requests
            user_pending_count = sum(1 for req in self.queues[channel_id] if req['user_id'] == user_id)
            if user_pending_count >= 2:  # Limit to 2 pending requests per user
                return False
            
            # Add request to queue
            request = {
                'id': len(self.queues[channel_id]) + int(current_time),
                'timestamp': current_time,
                'message': message,
                'content': content,
                'guild': guild,
                'attachments': attachments,
                'user_name': user_name,
                'is_dm': is_dm,
                'user_id': user_id,
                'reply_to_name': reply_to_name
            }
            
            self.queues[channel_id].append(request)
            
            # Start processing if not already processing
            if not self.processing[channel_id]:
                asyncio.create_task(self._process_queue(channel_id))
            
            return True
    
    async def _process_queue(self, channel_id: int):
        """Process all requests in the queue for a channel"""
        async with self.locks[channel_id]:
            if self.processing[channel_id]:
                return
            self.processing[channel_id] = True
        
        try:
            while self.queues[channel_id]:
                # Get the next request
                async with self.locks[channel_id]:
                    if not self.queues[channel_id]:
                        break
                    request = self.queues[channel_id].pop(0)
                
                # Process the request
                await self._process_single_request(channel_id, request)
                
                # Small delay between requests to prevent overwhelming
                await asyncio.sleep(0.5)
                
        finally:
            async with self.locks[channel_id]:
                self.processing[channel_id] = False
    
    async def _process_single_request(self, channel_id: int, request: dict):
        """Process a single request with proper context"""
        # print(f"DEBUG: _process_single_request called for channel {channel_id}, content={repr(request.get('content'))}")
        try:
            message = request['message']
            content = request['content']
            guild = request['guild']
            attachments = request['attachments']
            user_name = request['user_name']
            is_dm = request['is_dm']
            user_id = request['user_id']
            reply_to_name = request.get('reply_to_name')
            
            async with message.channel.typing():
                # LOAD CHANNEL HISTORY FROM DISCORD
                # Always load fresh history for server channels to ensure full context
                guild_id = guild.id if guild else None
                
                # Load history from Discord for all server channels (not DMs)
                # This ensures the bot always has the complete conversation context
                # Pass the triggering message so it can be excluded from the history load
                if not is_dm:
                    await load_channel_history_from_discord(message.channel, guild, channel_id, exclude_message_id=message.id)
                
                # Add the user's message to history
                await add_to_history(channel_id, "user", content, user_id, guild.id if guild else None, attachments, user_name, reply_to=reply_to_name)

                # Check if the last message in history is from the assistant
                current_history = get_conversation_history(channel_id)
                last_message_is_assistant = current_history and current_history[-1]["role"] == "assistant"
                
                # If the last message was from assistant, add continuation prompt
                if last_message_is_assistant:
                    await add_to_history(
                        channel_id, 
                        "user", 
                        "[Continue the conversation naturally from where you left off.]",
                        user_id=None,
                        guild_id=guild.id if guild else None,
                        user_name=None
                    )
                
                # Generate response using the main generate_response function (includes debug logging)
                bot_response = await generate_response(
                    channel_id=channel_id,
                    user_message=content,
                    guild=guild,
                    attachments=attachments,
                    user_name=user_name,
                    is_dm=is_dm,
                    user_id=user_id,
                    original_message=message
                )

                if bot_response is None:
                    return

                # STORE THE ORIGINAL RESPONSE WITH REACTIONS FOR HISTORY
                original_response_with_reactions = bot_response

                # PROCESS REACTIONS FIRST (this removes [REACT: X] from the response)
                if message:
                    bot_response = await process_and_add_reactions(bot_response, message)

                # THEN CLEAN EMOJIS (after reactions are processed)
                if bot_response and guild:
                    bot_response = clean_malformed_emojis(bot_response, guild)

                # CLEAN EM-DASHES (after emojis are cleaned)
                if bot_response:
                    bot_response = clean_em_dashes(bot_response)

                # CLEAN BOT NAME PREFIX (remove persona name from output)
                if bot_response:
                    bot_response = clean_bot_name_prefix(bot_response, guild.id if guild else None, user_id, is_dm)

                # Assistant response is already added to history in generate_response()
                
                # Finally sanitize user mentions
                if bot_response and not bot_response.startswith("❌"):
                    bot_response = sanitize_user_mentions(bot_response, guild)

                if bot_response is None:
                    return
                
                # Add a small delay to make responses feel more human-like
                await asyncio.sleep(1.0)
                
                # Check if the response is an error that should be temporary
                is_temp_error = bot_response.startswith("[TEMP_ERROR]")
                if is_temp_error:
                    bot_response = bot_response.replace("[TEMP_ERROR] ", "")
                    # Send as dismissible error instead of regular message
                    try:
                        await send_dismissible_error(message.channel, message.author, bot_response)
                        print(f"Sent dismissible error response: {bot_response[:100]}...")
                        return  # Don't continue with normal message sending
                    except Exception as send_error:
                        print(f"Failed to send dismissible error response: {send_error}")
                        # Continue with normal message sending as fallback
                
                # Send the response
                message_parts = split_message_by_newlines(bot_response)
                is_dm = isinstance(message.channel, discord.DMChannel)
                use_reply = not is_dm and not message.author.bot
                
                sent_messages = []
                if len(message_parts) > 1:
                    for part in message_parts:
                        if len(part) > 4000:
                            for i in range(0, len(part), 4000):
                                if use_reply:
                                    sent_msg = await message.reply(part[i:i+4000], delete_after=15.0 if is_temp_error else None)
                                else:
                                    sent_msg = await message.channel.send(part[i:i+4000], delete_after=15.0 if is_temp_error else None)
                                sent_messages.append(sent_msg)
                        else:
                            if use_reply:
                                sent_msg = await message.reply(part, delete_after=15.0 if is_temp_error else None)
                            else:
                                sent_msg = await message.channel.send(part, delete_after=15.0 if is_temp_error else None)
                            sent_messages.append(sent_msg)
                elif bot_response:
                    if len(bot_response) > 4000:
                        for i in range(0, len(bot_response), 4000):
                            if use_reply:
                                sent_msg = await message.reply(bot_response[i:i+4000], delete_after=15.0 if is_temp_error else None)
                            else:
                                sent_msg = await message.channel.send(bot_response[i:i+4000], delete_after=15.0 if is_temp_error else None)
                            sent_messages.append(sent_msg)
                    else:
                        if use_reply:
                            sent_msg = await message.reply(bot_response, delete_after=15.0 if is_temp_error else None)
                        else:
                            sent_msg = await message.channel.send(bot_response, delete_after=15.0 if is_temp_error else None)
                        sent_messages.append(sent_msg)
                
                if len(sent_messages) > 1:
                    store_multipart_response(message.channel.id, [msg.id for msg in sent_messages], bot_response)
                        
        except Exception as e:
            print(f"Error processing request: {e}")
            try:
                error_msg = f"❌ Sorry, I encountered an error processing your request: {str(e)}"
                # Truncate error message to stay under Discord's 4000 character limit
                if len(error_msg) > 3950:  # Leave some buffer
                    error_msg = error_msg[:3950] + "..."
                
                # Check if this is a Discord API error that should be ephemeral/temporary
                is_api_error = ("400 Bad Request" in str(e) or 
                               "error code" in str(e) or 
                               "50035" in str(e) or
                               "Invalid Form Body" in str(e))
                
                if is_api_error:
                    # Send as dismissible error message
                    try:
                        await send_dismissible_error(message.channel, message.author, error_msg)
                        print(f"Sent dismissible error message: {error_msg[:100]}...")
                    except Exception as send_error:
                        print(f"Failed to send dismissible error message: {send_error}")
                        # Fallback to regular temporary message
                        try:
                            temp_msg = await message.channel.send(error_msg, delete_after=15.0)
                        except Exception as fallback_error:
                            print(f"Failed to send fallback error message: {fallback_error}")
                else:
                    # Send regular error message
                    try:
                        await message.channel.send(error_msg)
                    except Exception as send_error:
                        print(f"Failed to send error message: {send_error}")
                        
            except Exception as inner_e:
                print(f"Error in error handling: {inner_e}")
                try:
                    fallback_msg = "❌ Sorry, I encountered an error processing your request."
                    await message.channel.send(fallback_msg)
                except Exception as fallback_error:
                    print(f"Failed to send fallback error message: {fallback_error}")

# Initialize the request queue
request_queue = RequestQueue()

class AutonomousManager:
    """Manages autonomous response behavior settings per channel"""
    def __init__(self):
        self.settings: Dict[int, Dict[int, Dict[str, any]]] = {}
        self.load_data()
    
    def set_channel_autonomous(self, guild_id: int, channel_id: int, enabled: bool, chance: int = 10):
        """Configure autonomous behavior for a channel"""
        if guild_id not in self.settings:
            self.settings[guild_id] = {}
        
        self.settings[guild_id][channel_id] = {
            "enabled": enabled,
            "chance": max(1, min(100, chance))
        }
        self.save_data()
    
    def get_channel_settings(self, guild_id: int, channel_id: int) -> Dict[str, any]:
        """Get autonomous settings for a channel"""
        return self.settings.get(guild_id, {}).get(channel_id, {"enabled": False, "chance": 10})
    
    def should_respond_autonomously(self, guild_id: int, channel_id: int) -> bool:
        """Determine if bot should respond autonomously"""
        settings = self.get_channel_settings(guild_id, channel_id)
        if not settings["enabled"]:
            return False
        return random.randint(1, 100) <= settings["chance"]
    
    def list_autonomous_channels(self, guild_id: int) -> Dict[int, Dict[str, any]]:
        """List all autonomous channels for a guild"""
        return self.settings.get(guild_id, {})
    
    def save_data(self):
        """Persist autonomous settings to file"""
        try:
            json_data = {}
            for guild_id, guild_settings in self.settings.items():
                json_data[str(guild_id)] = {str(channel_id): settings for channel_id, settings in guild_settings.items()}
            
            with open(AUTONOMOUS_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass
    
    def load_data(self):
        """Load autonomous settings from file"""
        try:
            if os.path.exists(AUTONOMOUS_FILE):
                with open(AUTONOMOUS_FILE, 'r') as f:
                    json_data = json.load(f)
                
                self.settings = {}
                for guild_id_str, guild_settings in json_data.items():
                    guild_id = int(guild_id_str)
                    self.settings[guild_id] = {int(channel_id_str): settings for channel_id_str, settings in guild_settings.items()}
        except Exception:
            self.settings = {}

# Utility functions for data persistence
def save_json_data(file_path: str, data: dict, convert_keys=True):
    """Generic function to save dictionary data to JSON file"""
    try:
        if convert_keys:
            json_data = {str(k): v for k, v in data.items()}
        else:
            json_data = data
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    except Exception:
        pass

def load_json_data(file_path: str, convert_keys=True) -> dict:
    """Generic function to load dictionary data from JSON file"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            if convert_keys:
                return {int(k): v for k, v in data.items()}
            return data
    except Exception:
        pass
    return {}

def save_personalities():
    """Save personality settings to file"""
    save_data = {
        "guild_personalities": guild_personalities,
        "custom_personalities": custom_personalities
    }
    save_json_data(PERSONALITIES_FILE, save_data, convert_keys=False)

def load_personalities():
    """Load personality settings from file"""
    data = load_json_data(PERSONALITIES_FILE, convert_keys=False)
    guild_perss = {int(k): v for k, v in data.get("guild_personalities", {}).items()}
    custom_perss = {int(k): v for k, v in data.get("custom_personalities", {}).items()}
    return guild_perss, custom_perss

def get_shared_guild(user_id: int) -> discord.Guild:
    """Get a guild that both the bot and user are members of"""
    user = client.get_user(user_id)
    if not user:
        return None
    
    for guild in client.guilds:
        member = guild.get_member(user_id)
        if member:
            return guild
        
        if user in guild.members:
            return guild
    
    return None

async def get_shared_guild_async(user_id: int) -> discord.Guild:
    """Async version that can fetch member if not in cache"""
    for guild in client.guilds:
        try:
            member = await guild.fetch_member(user_id)
            if member:
                return guild
        except (discord.NotFound, discord.Forbidden, Exception):
            continue
    
    return None

def get_shared_server_settings(user_id: int) -> tuple:
    """Get settings from a shared server for DM conversations"""
    shared_guild = get_shared_guild(user_id)
    if shared_guild:
        return shared_guild.id, shared_guild
    return None, None

def check_admin_permissions(interaction: discord.Interaction) -> bool:
    """Check if user has administrator permissions"""
    if not interaction.guild:
        return False
    return interaction.user.guild_permissions.administrator

def convert_emojis_to_simple(text: str) -> str:
    """Convert full Discord emoji format <:name:id> to simple :name: format for AI learning"""
    
    # Pattern for animated and static emojis: <a:name:id> or <:name:id>
    emoji_pattern = r'<a?:([a-zA-Z0-9_]+):\d+>'
    
    def replace_emoji(match):
        emoji_name = match.group(1)
        return f":{emoji_name}:"
    
    return re.sub(emoji_pattern, replace_emoji, text)

def remove_thinking_tags(text: str) -> str:
    """Remove thinking tags and their content from AI responses"""
    if not text:
        return text
    
    # Remove thinking tags with various formats: <thinking>, <think>, etc.
    # This regex removes the opening tag, content, and closing tag
    thinking_pattern = r'<(\w*think\w*)[^>]*>.*?</\1>'
    text = re.sub(thinking_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Also remove self-closing thinking tags
    text = re.sub(r'<(\w*think\w*)[^>]*/>', '', text, flags=re.IGNORECASE)
    
    # Also remove stray/orphan closing tags (e.g. </think>, </thinking>)
    text = re.sub(r'.*?</\w*think\w*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text
    return text

def clean_malformed_emojis(text: str, guild: discord.Guild = None) -> str:
    """Convert :emoji_name: format to proper Discord format or remove invalid ones"""
    
    if not text:
        return text
    
    # First, fix any double-wrapped emojis like <<:emoji:id>> or <<a:emoji:id>>
    text = re.sub(r'<(<a?:[a-zA-Z0-9_]+:[0-9]+>)>', r'\1', text)
    
    # Store valid Discord emojis to protect them during cleaning
    valid_emojis = []
    emoji_placeholders = []
    
    # Find and temporarily replace all valid Discord emojis with placeholders
    valid_emoji_pattern = r'<(a?):([a-zA-Z0-9_]+):([0-9]+)>'
    matches = list(re.finditer(valid_emoji_pattern, text))
    
    for i, match in enumerate(matches):
        placeholder = f"__EMOJI_PLACEHOLDER_{i}__"
        valid_emojis.append(match.group(0))  # Store the full emoji
        emoji_placeholders.append(placeholder)
        text = text.replace(match.group(0), placeholder, 1)
    
    # Pattern to match :emoji_name: format (simple Discord emoji syntax)
    simple_emoji_pattern = r':([a-zA-Z0-9_]+):'
    
    def replace_emoji(match):
        full_match = match.group(0)
        emoji_name = match.group(1).lower()
        
        # Check if this is already part of a placeholder (skip it)
        if "__EMOJI_PLACEHOLDER_" in full_match:
            return full_match
        
        # If we have a guild, try to find the actual emoji (both animated and static)
        if guild:
            for emoji in guild.emojis:
                if emoji.name.lower() == emoji_name:
                    return f"<{'a' if emoji.animated else ''}:{emoji.name}:{emoji.id}>"
        
        # Check if it might be a standard Unicode emoji name
        common_unicode_emojis = {
            'smile', 'grin', 'joy', 'heart', 'thumbsup', 'thumbsdown', 
            'fire', 'star', 'eyes', 'thinking', 'shrug', 'wave', 'clap'
        }
        
        if emoji_name in common_unicode_emojis:
            return f":{emoji_name}:"
        
        # Remove unknown emoji
        return ""
    
    # Replace :emoji_name: patterns (but not placeholders)
    cleaned_text = re.sub(simple_emoji_pattern, replace_emoji, text)
    
    # Only clean up ACTUALLY malformed patterns (be more specific)
    leftover_patterns = [
        r'<a?:[a-zA-Z0-9_]*$',           # Incomplete at end of string
        r'<a?:[a-zA-Z0-9_]*:[0-9]*$',    # Missing closing bracket at end
        r'<a?:[a-zA-Z0-9_]+:$',          # Missing ID and closing bracket
        r'<a?:$',                        # Just opening
    ]
    
    for pattern in leftover_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text)
    
    # Restore the valid emojis from placeholders
    for placeholder, original_emoji in zip(emoji_placeholders, valid_emojis):
        cleaned_text = cleaned_text.replace(placeholder, original_emoji)
    
    # Clean up multiple spaces but preserve line structure
    lines = cleaned_text.split('\n')
    cleaned_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    cleaned_text = '\n'.join(cleaned_lines)
    
    return cleaned_text

def clean_em_dashes(text: str) -> str:
    """Replace em-dashes with appropriate punctuation based on context.
    
    - Mid-sentence em-dashes (text before and after) become ", "
    - End-sentence em-dashes (text before, nothing after) become "-"
    """
    if not text:
        return text
    
    # Pattern for em-dash with text before and after (mid-sentence)
    # Lookbehind: non-whitespace character before em-dash
    # Lookahead: non-whitespace character after em-dash
    mid_sentence_pattern = r'(?<=\S)—(?=\S)'
    
    # Replace mid-sentence em-dashes with ", "
    text = re.sub(mid_sentence_pattern, ", ", text)
    
    # Pattern for em-dash at end of sentence (text before, but nothing after except whitespace/punctuation)
    # Lookbehind: non-whitespace character before em-dash
    # Lookahead: whitespace, punctuation, or end of string
    end_sentence_pattern = r'(?<=\S)—(?=\s|$|[.!?])'
    
    # Replace end-sentence em-dashes with "-"
    text = re.sub(end_sentence_pattern, "-", text)
    
    return text

def clean_bot_name_prefix(text: str, guild_id: int = None, user_id: int = None, is_dm: bool = False) -> str:
    """Remove bot persona name prefix from the response text before sending to Discord"""
    if not text:
        return text
    
    # Get the bot's persona name
    bot_name = get_bot_persona_name(guild_id, user_id, is_dm)
    
    # Remove the prefix if it exists
    if text.startswith(f"{bot_name}: "):
        return text[len(f"{bot_name}: "):]
    elif text.startswith(f"{bot_name}:"):
        return text[len(f"{bot_name}:"):]
    
    return text

def save_custom_prompts():
    """Save custom prompts to file"""
    save_data = {
        "custom_prompts": {str(guild_id): prompts for guild_id, prompts in custom_prompts.items()}
    }
    save_json_data(CUSTOM_PROMPTS_FILE, save_data, convert_keys=False)

def save_custom_format_instructions():
    """Save custom format instructions to file"""
    save_json_data(CUSTOM_FORMAT_INSTRUCTIONS_FILE, custom_format_instructions, convert_keys=False)

def load_custom_prompts():
    """Load custom prompts from file"""
    data = load_json_data(CUSTOM_PROMPTS_FILE, convert_keys=False)
    custom_prompts_data = {}
    if "custom_prompts" in data:
        for guild_id_str, prompts in data["custom_prompts"].items():
            custom_prompts_data[int(guild_id_str)] = prompts
    return custom_prompts_data

# Remove all custom prompt related variables and functions

# Global state variables
conversations: Dict[int, List[Dict]] = {}

def clean_conversation_history():
    """Clean any complex content (with base64 images) from conversation history, keeping only text"""
    for channel_id in conversations:
        for message in conversations[channel_id]:
            content = message.get("content", "")
            if isinstance(content, list):
                # Extract only text parts from complex content
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                message["content"] = " ".join(text_parts).strip()

# Clean conversation history on startup
clean_conversation_history()

channel_format_settings: Dict[int, str] = load_json_data(FORMAT_SETTINGS_FILE)
dm_format_settings: Dict[int, str] = load_json_data(DM_FORMAT_SETTINGS_FILE)
server_format_defaults: Dict[int, str] = load_json_data(SERVER_FORMAT_DEFAULTS_FILE)
guild_nsfw_settings: Dict[int, bool] = load_json_data(NSFW_SETTINGS_FILE)
dm_nsfw_settings: Dict[int, bool] = load_json_data(DM_NSFW_SETTINGS_FILE)
custom_format_instructions: Dict[str, str] = load_json_data(CUSTOM_FORMAT_INSTRUCTIONS_FILE, convert_keys=False)
prefill_settings: Dict[int, str] = load_json_data(PREFILL_SETTINGS_FILE)
multipart_responses: Dict[int, Dict[int, Tuple[List[int], str]]] = {}
multipart_response_counter: Dict[int, int] = {}
guild_personalities: Dict[int, str] = {}
custom_personalities: Dict[int, Dict[str, Dict[str, str]]] = {}
guild_history_lengths: Dict[int, int] = load_json_data(HISTORY_LENGTHS_FILE)
recent_participants: Dict[int, Set[int]] = {}
custom_activity: str = load_json_data(ACTIVITY_FILE, convert_keys=False).get("custom_activity", "")
guild_temperatures: Dict[int, float] = load_json_data(TEMPERATURE_FILE)
welcome_dm_sent: Dict[int, bool] = load_json_data(WELCOME_SENT_FILE)
dm_server_selection: Dict[int, int] = load_json_data(DM_SERVER_SELECTION_FILE)
guild_dm_enabled: Dict[int, bool] = load_json_data(DM_ENABLED_FILE)
bot_persona_name: str = "Assistant"
recently_deleted_dm_messages: Dict[int, Set[int]] = {}

# Temporary old system variables - TO BE COMPLETELY REMOVED
# custom_prompts: Dict[int, Dict[str, Dict[str, str]]] = {}
# channel_prompt_settings: Dict[int, str] = {}
# dm_prompt_settings: Dict[int, str] = {}

# Migration function removed - no longer needed

# Run migration on startup
# migrate_old_nsfw_styles()  # Removed - no longer needed

# Initialize managers
lore_book = LoreBook()
autonomous_manager = AutonomousManager()
memory_manager = MemoryManager()
dm_manager = DMManager()

# Load personality data
loaded_guild_personalities, loaded_custom_personalities = load_personalities()
guild_personalities = loaded_guild_personalities
custom_personalities = loaded_custom_personalities

# Default personality configuration
DEFAULT_PERSONALITIES = {
    "default": {
        "name": "Assistant",
        "prompt": "A helpful AI assistant. Friendly, curious, and engaging in conversations."
    }
}

def get_bot_persona_name(guild_id: int = None, user_id: int = None, is_dm: bool = False) -> str:
    """Get the bot's current persona name based on context"""
    global bot_persona_name
    
    # For DMs, get persona from selected server or shared guild
    if is_dm and user_id:
        # Check if user has a specific DM personality set
        if user_id in dm_manager.dm_personalities:
            preferred_guild_id, preferred_personality = dm_manager.dm_personalities[user_id]
            guild_id = preferred_guild_id
            personality_name = preferred_personality
        else:
            # Use selected server or shared guild
            selected_guild_id = dm_server_selection.get(user_id)
            if selected_guild_id:
                guild_id = selected_guild_id
                personality_name = guild_personalities.get(guild_id, "default")
            else:
                shared_guild = get_shared_guild(user_id)
                if shared_guild:
                    guild_id = shared_guild.id
                    personality_name = guild_personalities.get(guild_id, "default")
                else:
                    personality_name = "default"
    else:
        # Server context
        personality_name = guild_personalities.get(guild_id, "default") if guild_id else "default"
    
    # Extract the name from personality data
    if guild_id and guild_id in custom_personalities and personality_name in custom_personalities[guild_id]:
        return custom_personalities[guild_id][personality_name]["name"]
    else:
        return DEFAULT_PERSONALITIES["default"]["name"]

def update_bot_persona_name(guild_id: int = None, user_id: int = None, is_dm: bool = False):
    """Update the global bot persona name based on current context"""
    global bot_persona_name
    bot_persona_name = get_bot_persona_name(guild_id, user_id, is_dm)

guild_emoji_cache: Dict[int, Dict[str, str]] = {}


def get_guild_emoji_lookup(guild: discord.Guild) -> Dict[str, str]:
    """Build a lookup map for quick guild emoji name resolution."""
    if not guild:
        return {}

    cached_lookup = guild_emoji_cache.get(guild.id)
    if cached_lookup is not None and len(cached_lookup) == len(guild.emojis):
        return cached_lookup

    lookup = {}
    for emoji in guild.emojis:
        lookup[emoji.name.lower()] = f"<{'a' if emoji.animated else ''}:{emoji.name}:{emoji.id}>"

    guild_emoji_cache[guild.id] = lookup
    return lookup


def get_guild_emojis(guild: discord.Guild) -> str:
    """Get formatted list of guild emojis for system prompt with simple :name: format"""
    if not guild:
        return ""
    
    # Get both animated and non-animated emojis
    available_emojis = list(guild.emojis)  # This includes both animated and static
    
    if available_emojis:
        # Select up to 50 emojis (mix of animated and static)
        max_emojis = min(50, len(available_emojis))
        
        # If we have more emojis than the limit, prioritize by usage and variety
        if len(available_emojis) > max_emojis:
            # Sort by available status first (prioritize available emojis)
            available_emojis = sorted(available_emojis, key=lambda e: e.available, reverse=True)
            
            # Try to get a good mix of animated and static emojis
            static_emojis = [e for e in available_emojis if not e.animated]
            animated_emojis = [e for e in available_emojis if e.animated]
            
            selected_emojis = []
            
            # Take up to 35 static emojis and 15 animated emojis for variety
            static_count = min(35, len(static_emojis), max_emojis)
            animated_count = min(15, len(animated_emojis), max_emojis - static_count)
            
            if static_emojis:
                selected_emojis.extend(random.sample(static_emojis, static_count))
            if animated_emojis: 
                selected_emojis.extend(random.sample(animated_emojis, animated_count))
            
            # If we still need more emojis, fill from remaining
            remaining_needed = max_emojis - len(selected_emojis)
            if remaining_needed > 0:
                remaining_emojis = [e for e in available_emojis if e not in selected_emojis]
                if remaining_emojis:
                    selected_emojis.extend(random.sample(remaining_emojis, min(remaining_needed, len(remaining_emojis))))
        else:
            selected_emojis = available_emojis
        
        # Sort by name for consistent display
        selected_emojis.sort(key=lambda e: e.name.lower())
        
        # Format them in simple :name: format
        emoji_list = ' '.join([f":{emoji.name}:" for emoji in selected_emojis])
        
        return f"\n\n<emojis>Available custom emojis for this server:\n{emoji_list}\nTEMPLATE: When using custom emojis follow the exact format of :emoji: in your responses, otherwise they won't work! Limit their usage.\nREACTIONS: You can also react to the users' messages with emojis! To add a reaction, include [REACT: emoji] anywhere in your response. Examples: [REACT: 😄] or [REACT: :custom_emoji:]. Occasionally, react to show emotion, agreement, humor, or acknowledgment.</emojis>"
    return ""

def get_guild_emoji_list(guild: discord.Guild) -> str:
    """Get simple comma-separated list of guild emojis for system prompt
    
    Selects up to 50 emojis from the server, prioritizing:
    - Available emojis over unavailable ones
    - A mix of static and animated emojis (roughly 35 static, 15 animated)
    - Alphabetical order for consistency
    """
    if not guild:
        return "No custom emojis available."
    
    # Get both animated and non-animated emojis
    available_emojis = list(guild.emojis)
    
    if not available_emojis:
        return "No custom emojis available for this server."
    
    # Select up to 50 emojis (mix of animated and static)
    max_emojis = min(50, len(available_emojis))
    
    # If we have more emojis than the limit, prioritize by usage and variety
    if len(available_emojis) > max_emojis:
        # Sort by available status first (prioritize available emojis)
        available_emojis = sorted(available_emojis, key=lambda e: e.available, reverse=True)
        
        # Try to get a good mix of animated and static emojis
        static_emojis = [e for e in available_emojis if not e.animated]
        animated_emojis = [e for e in available_emojis if e.animated]
        
        selected_emojis = []
        
        # Take up to 35 static emojis and 15 animated emojis for variety
        static_count = min(35, len(static_emojis), max_emojis)
        animated_count = min(15, len(animated_emojis), max_emojis - static_count)
        
        if static_emojis:
            selected_emojis.extend(random.sample(static_emojis, static_count))
        if animated_emojis:
            selected_emojis.extend(random.sample(animated_emojis, animated_count))
        
        # If we still need more emojis, fill from remaining
        remaining_needed = max_emojis - len(selected_emojis)
        if remaining_needed > 0:
            remaining_emojis = [e for e in available_emojis if e not in selected_emojis]
            if remaining_emojis:
                selected_emojis.extend(random.sample(remaining_emojis, min(remaining_needed, len(remaining_emojis))))
    else:
        selected_emojis = available_emojis
    
    # Sort by name for consistent display
    selected_emojis.sort(key=lambda e: e.name.lower())
    
    # Format them in simple :name: format
    emoji_count = len(selected_emojis)
    total_count = len(available_emojis)
    emoji_list = ' '.join([f":{emoji.name}:" for emoji in selected_emojis])
    
    # Add a note if we're showing a subset
    if emoji_count < total_count:
        emoji_list += f" (showing {emoji_count} of {total_count} available emojis)"
    
    return emoji_list

async def process_and_add_reactions(bot_response: str, user_message: discord.Message) -> str:
    """Process bot response for reaction instructions and add reactions to user message"""
    if not bot_response:
        return bot_response

    # Find all reaction instructions in the response
    reaction_pattern = r'\[REACT:\s*([^\]]+)\]'
    reactions = re.findall(reaction_pattern, bot_response)
    
    # Remove reaction instructions from the response
    if reactions:
        cleaned_response = re.sub(reaction_pattern, ' ', bot_response)
        cleaned_response = re.sub(r'  +', ' ', cleaned_response).strip()
    else:
        cleaned_response = bot_response
    
    # Add reactions if we have a message to react to
    if user_message is not None and reactions:
        for reaction in reactions:
            reaction = reaction.strip()
            
            # Convert custom emoji format if needed
            converted_reaction = convert_emoji_for_reaction(reaction, user_message.guild)
            
            # Skip if the emoji was filtered out (from another server)
            if converted_reaction is None:
                continue
            
            try:
                await user_message.add_reaction(converted_reaction)
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
            except discord.HTTPException as e:
                # If it still fails, just skip it
                # print(f"Failed to add reaction '{reaction}': {e}")
                continue
            except Exception as e:
                print(f"Unexpected error adding reaction '{reaction}': {e}")
                continue
    
    return cleaned_response

def convert_emoji_for_reaction(emoji_text: str, guild: discord.Guild = None) -> str:
    """Convert emoji text to proper format for reactions with improved matching"""
    
    if emoji_text.startswith('<:') or emoji_text.startswith('<a:'):
        if guild:
            emoji_id_match = re.search(r':(\d+)>', emoji_text)
            if emoji_id_match:
                emoji_id = int(emoji_id_match.group(1))
                if any(emoji.id == emoji_id for emoji in guild.emojis):
                    return emoji_text
                return None
        return None
    
    if emoji_text.startswith(':') and emoji_text.endswith(':') and guild:
        emoji_name = emoji_text.strip(':')
        mapped_emoji = get_guild_emoji_lookup(guild).get(emoji_name.lower())
        if mapped_emoji:
            return mapped_emoji

        unicode_emoji_map = {
            'smile': '😊', 'heart': '❤️', 'thumbsup': '👍', 'thumbsdown': '👎',
            'fire': '🔥', 'star': '⭐', 'eyes': '👀', 'thinking': '🤔',
            'shrug': '🤷', 'wave': '👋', 'clap': '👏', 'kiss': '😘',
            'hug': '🤗', 'laugh': '😂', 'cry': '😢', 'angry': '😠',
            'mad': '😡', 'love': '💕'
        }
        
        if emoji_name.lower() in unicode_emoji_map:
            return unicode_emoji_map[emoji_name.lower()]
        
        return None
    
    return emoji_text

def estimate_message_size(messages: List[Dict], system_prompt: str) -> int:
    """Estimate the total size of the request in characters/tokens"""
    total_size = len(system_prompt)
    
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total_size += len(content)
        elif isinstance(content, list):
            # Handle complex content with images
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        total_size += len(part.get("text", ""))
                    elif part.get("type") == "image":
                        # Estimate image contributes roughly equivalent to 1000 characters
                        total_size += 1000
                    elif part.get("type") == "image_url":
                        # OpenAI format images
                        total_size += 1000
    
    return total_size

def is_supported_file_type(filename: str) -> bool:
    """Check if file type is supported for processing"""
    supported_image_types = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    supported_audio_types = ['.mp3', '.wav', '.ogg', '.m4a', '.webm']
    supported_text_types = ['.txt', '.md', '.json', '.csv', '.log']
    
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in 
               supported_image_types + supported_audio_types + supported_text_types)

def get_attachment_size_limit(attachment: discord.Attachment, provider: str = "claude") -> int:
    """Get size limit based on attachment type and provider"""
    filename_lower = attachment.filename.lower()
    
    # Image files
    if any(filename_lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
        return 20 * 1024 * 1024 if provider == "openai" else 8 * 1024 * 1024  # 20MB for OpenAI, 8MB for others
    
    # Audio files for voice processing
    elif any(filename_lower.endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
        return 25 * 1024 * 1024  # 25MB limit for audio files
    
    # Text files (can be included in context)
    elif any(filename_lower.endswith(ext) for ext in ['.txt', '.md', '.json', '.csv', '.log']):
        return 1 * 1024 * 1024  # 1MB limit for text files
    
    # All other files - very restrictive
    else:
        return 100 * 1024  # 100KB limit for other file types

def should_process_attachment(attachment: discord.Attachment, provider: str = "claude") -> Tuple[bool, str]:
    """Determine if attachment should be processed and return reason if not"""
    filename_lower = attachment.filename.lower()
    
    # Check for unsupported/problematic file types
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    large_doc_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']
    archive_extensions = ['.zip', '.rar', '.7z', '.tar', '.gz']
    executable_extensions = ['.exe', '.msi', '.dmg', '.app', '.deb', '.rpm']
    
    # Block video files (too large and not useful for text AI)
    if any(filename_lower.endswith(ext) for ext in video_extensions):
        return False, f"Video files ({attachment.filename}) are not supported to prevent request size issues"
    
    # Block large document formats that require special processing
    if any(filename_lower.endswith(ext) for ext in large_doc_extensions):
        return False, f"Document files ({attachment.filename}) are not supported - please convert to text format"
    
    # Block archives and executables
    if any(filename_lower.endswith(ext) for ext in archive_extensions + executable_extensions):
        return False, f"Archive/executable files ({attachment.filename}) are not supported"
    
    # Check if file type is supported
    if not is_supported_file_type(attachment.filename):
        return False, f"File type not supported ({attachment.filename})"
    
    # Check file size
    size_limit = get_attachment_size_limit(attachment, provider)
    if attachment.size > size_limit:
        size_limit_mb = size_limit / (1024 * 1024)
        actual_size_mb = attachment.size / (1024 * 1024)
        return False, f"File too large ({attachment.filename}: {actual_size_mb:.1f}MB, limit: {size_limit_mb:.1f}MB)"
    
    return True, ""

async def process_image_attachment(attachment: discord.Attachment, provider: str = "claude") -> dict:
    """Process image attachment and return provider-specific format"""
    try:
        if not any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
            return None
        
        # Use the new unified size checking
        should_process, reason = should_process_attachment(attachment, provider)
        if not should_process:
            print(f"Skipping image: {reason}")
            return None
            
        # print(f"Processing image: {attachment.filename} for provider: {provider}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                if resp.status == 200:
                    image_data = await resp.read()
                    # print(f"Downloaded image data: {len(image_data)} bytes")
                    
                    media_type_map = {
                        '.png': "image/png",
                        '.gif': "image/gif", 
                        '.webp': "image/webp"
                    }
                    
                    file_ext = next((ext for ext in media_type_map.keys() if attachment.filename.lower().endswith(ext)), None)
                    media_type = media_type_map.get(file_ext, "image/jpeg")
                    
                    # print(f"Detected media type: {media_type}")
                    
                    if provider in ["openai", "custom"]:
                        # OpenAI format - for custom providers (including OpenRouter)
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        result = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}",
                                "detail": "high"  # Changed from "auto" to "high"
                            }
                        }
                        # print(f"Created OpenAI format image for custom provider")
                        return result
                    elif provider == "gemini":
                        # Gemini format
                        result = {
                            "type": "image",
                            "data": base64.b64encode(image_data).decode('utf-8'),
                            "media_type": media_type
                        }
                        # print(f"Created Gemini format image")
                        return result
                    else:
                        # Claude format
                        result = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(image_data).decode('utf-8')
                            }
                        }
                        # print(f"Created Claude format image")
                        return result
                else:
                    # print(f"Failed to download image: HTTP {resp.status}")
                    return None
    except Exception as e:
        # print(f"Error processing image: {e}")
        return None

def get_conversation_history(channel_id: int) -> List[Dict]:
    """Get conversation history for a channel"""
    return conversations.get(channel_id, [])

def get_history_length(guild_id: int) -> int:
    """Get history length setting for a guild"""
    return guild_history_lengths.get(guild_id, 50)

def get_temperature(guild_id: int) -> float:
    """Get temperature setting for a guild"""
    return guild_temperatures.get(guild_id, 1.0)

def store_multipart_response(channel_id: int, message_ids: List[int], full_content: str):
    """Store a multi-part response for tracking"""
    if channel_id not in multipart_responses:
        multipart_responses[channel_id] = {}
        multipart_response_counter[channel_id] = 0
    
    multipart_response_counter[channel_id] += 1
    response_id = multipart_response_counter[channel_id]
    multipart_responses[channel_id][response_id] = (message_ids, full_content)

def find_multipart_response(channel_id: int, message_id: int) -> Tuple[int, List[int], str]:
    """Find which multipart response a message belongs to"""
    if channel_id not in multipart_responses:
        return None, [], ""
    
    for response_id, (message_ids, full_content) in multipart_responses[channel_id].items():
        if message_id in message_ids:
            return response_id, message_ids, full_content
    
    return None, [], ""

async def get_bot_last_logical_message(channel) -> Tuple[List[discord.Message], str]:
    """Get the bot's last logical message (may be multiple Discord messages)"""
    try:
        recent_bot_messages = []
        async for message in channel.history(limit=20):
            if message.author == client.user and len(message.content.strip()) > 0:
                recent_bot_messages.append(message)
                if len(recent_bot_messages) >= 10:
                    break
        
        if not recent_bot_messages:
            return [], ""
        
        # Check if the most recent bot message is part of a multipart response
        most_recent = recent_bot_messages[0]
        response_id, message_ids, full_content = find_multipart_response(channel.id, most_recent.id)
        
        if response_id:
            # This is part of a multipart response, get all messages
            all_messages = []
            for msg_id in message_ids:
                try:
                    msg = await channel.fetch_message(msg_id)
                    all_messages.append(msg)
                except Exception:
                    pass
            all_messages.sort(key=lambda m: m.created_at)
            return all_messages, full_content
        else:
            # Single message response
            return [most_recent], most_recent.content
    
    except Exception:
        return [], ""

async def add_to_history(channel_id: int, role: str, content: str, user_id: int = None, guild_id: int = None, attachments: List[discord.Attachment] = None, user_name: str = None, process_images: bool = True, reply_to: str = None) -> str:
    """Add a message to conversation history with proper formatting and image support"""
    # print(f"DEBUG: add_to_history called with role={role}, content={repr(content)}, user_name={repr(user_name)}, user_id={user_id}, guild_id={guild_id}")
    if channel_id not in conversations:
        conversations[channel_id] = []

    # Ensure content is not None
    if content is None:
        content = ""
    
    # Get guild object for mention conversion
    guild_obj = client.get_guild(guild_id) if guild_id else None
    
    # Track participants for lore activation (servers only)
    if user_id and role == "user" and guild_id:
        if channel_id not in recent_participants:
            recent_participants[channel_id] = set()
        recent_participants[channel_id].add(user_id)

    is_dm = not guild_id
    
    # Get the user/bot object to check if it's a bot
    user_obj = None
    if user_id and guild_id:
        guild_obj = client.get_guild(guild_id)
        if guild_obj:
            user_obj = guild_obj.get_member(user_id)
    elif user_id:
        user_obj = client.get_user(user_id)
    
    is_other_bot = user_obj and user_obj.bot and user_obj != client.user
    
   # Format user messages (including other bots treated as users)
    if role == "user" and user_name:
        if is_dm:
            if reply_to:
                formatted_content = f"[Replying to {reply_to}] {content}"
            else:
                formatted_content = content
        else:
            if is_other_bot:
                # For other bots, append all their messages to history as sent by the user
                # Convert any bot mentions to display names for clarity
                clean_content = convert_bot_mentions_to_names(content, guild_obj) if guild_id else content
                if reply_to:
                    formatted_content = f"{user_name}: [Replying to {reply_to}] {clean_content}"
                else:
                    formatted_content = f"{user_name}: {clean_content}"
            else:
                # Convert bot mentions to display names for better readability
                clean_content = convert_bot_mentions_to_names(content, guild_obj) if guild_id else content
                if reply_to:
                    formatted_content = f"{user_name}: [Replying to {reply_to}] {clean_content}"
                else:
                    formatted_content = f"{user_name}: {clean_content}"
    else:
        # For assistant messages, format with bot's persona name
        bot_name = get_bot_persona_name(guild_id, user_id, not guild_id)
        
        # Clean the AI response by removing any "NAME: " prefix if present
        clean_content = content
        if content.startswith(f"{bot_name}: "):
            clean_content = content[len(f"{bot_name}: "):]
        elif content.startswith(f"{bot_name}:"):
            clean_content = content[len(f"{bot_name}:"):]
        
        # Convert bot mentions to names and apply emoji conversion
        if guild_id:
            guild_obj = client.get_guild(guild_id)
            clean_content = convert_bot_mentions_to_names(clean_content, guild_obj) if guild_obj else clean_content
        clean_content = convert_emojis_to_simple(clean_content)
        
        formatted_content = f"{bot_name}: {clean_content}"

    # Ensure formatted_content is not None
    if formatted_content is None:
        formatted_content = content if content is not None else ""

    # Handle image attachments - create complex content for AI providers that support images
    message_content = formatted_content
    has_images = False
    
    if role == "user" and attachments and not is_other_bot and process_images:
        # Get current provider to determine image format
        provider_name = "claude"  # default
        
        if is_dm and user_id:
            # For DMs, get provider from selected server or shared guild
            selected_guild_id = dm_server_selection.get(user_id)
            if selected_guild_id:
                provider_name, _ = ai_manager.get_guild_settings(selected_guild_id)
            else:
                # Try to get from shared guild
                shared_guild = get_shared_guild(user_id)
                if shared_guild:
                    provider_name, _ = ai_manager.get_guild_settings(shared_guild.id)
        elif guild_id:
            # For servers, get provider directly
            provider_name, _ = ai_manager.get_guild_settings(guild_id)
        
        # Process images if provider supports them
        if provider_name in ["claude", "gemini", "openai", "custom"]:
            # Check if the current model supports vision
            supports_vision = False
            if provider_name == "claude":
                # Claude models generally support vision
                supports_vision = True
            elif provider_name == "gemini":
                # Gemini models support vision
                supports_vision = True
            elif provider_name == "openai":
                # Check specific OpenAI models
                current_model = None
                try:
                    if is_dm and user_id:
                        selected_guild_id = dm_server_selection.get(user_id)
                        if selected_guild_id:
                            _, current_model = ai_manager.get_guild_settings(selected_guild_id)
                        elif guild_id:
                            _, current_model = ai_manager.get_guild_settings(guild_id)
                    elif guild_id:
                        _, current_model = ai_manager.get_guild_settings(guild_id)
                    
                    if current_model:
                        vision_models = ["gpt-5", "gpt-5-chat-latest", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview", "gpt-4.1", "gpt-4.1-mini"]
                        supports_vision = any(vision_model in current_model.lower() for vision_model in vision_models)
                except:
                    supports_vision = False
            elif provider_name == "custom":
                # For custom providers, we check dynamically
                current_model = None
                try:
                    if is_dm and user_id:
                        selected_guild_id = dm_server_selection.get(user_id)
                        if selected_guild_id:
                            _, current_model = ai_manager.get_guild_settings(selected_guild_id)
                        elif guild_id:
                            _, current_model = ai_manager.get_guild_settings(guild_id)
                    elif guild_id:
                        _, current_model = ai_manager.get_guild_settings(guild_id)
                    
                    if current_model:
                        # Use the dynamic vision check for custom providers
                        custom_provider = ai_manager.providers.get("custom")
                        if custom_provider and hasattr(custom_provider, 'supports_vision_dynamic'):
                            supports_vision = asyncio.run(custom_provider.supports_vision_dynamic(current_model))
                        else:
                            supports_vision = False
                except:
                    supports_vision = False
            
            if supports_vision:
                image_parts = []
                text_parts = []
                
                # Add text content first
                if formatted_content.strip():
                    if provider_name == "openai":
                        text_parts.append({"type": "text", "text": formatted_content})
                    else:
                        text_parts.append({"type": "text", "text": formatted_content})
                
                # Process each attachment with comprehensive filtering
                total_processed_size = 0
                max_total_size = 50 * 1024 * 1024  # 50MB total limit for all attachments combined
                
                for attachment in attachments:
                    # Check if we should process this attachment
                    should_process, reason = should_process_attachment(attachment, provider_name)
                    
                    if not should_process:
                        # Add explanation for why attachment was skipped
                        text_parts.append({"type": "text", "text": f" [{reason}]"})
                        continue
                    
                    # Check total size limit
                    if total_processed_size + attachment.size > max_total_size:
                        text_parts.append({"type": "text", "text": f" [Attachment {attachment.filename} skipped - total size limit exceeded]"})
                        continue
                    
                    # Process based on file type
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        # Image processing
                        try:
                            image_data = await process_image_attachment(attachment, provider_name)
                            if image_data:
                                image_parts.append(image_data)
                                has_images = True
                                total_processed_size += attachment.size
                            else:
                                text_parts.append({"type": "text", "text": f" [Could not process image {attachment.filename}]"})
                        except Exception as e:
                            print(f"Error processing image {attachment.filename}: {e}")
                            text_parts.append({"type": "text", "text": f" [Error processing image {attachment.filename}]"})
                    
                    elif any(attachment.filename.lower().endswith(ext) for ext in ['.txt', '.md', '.json', '.csv', '.log']):
                        # Text file processing - read content if small enough
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(attachment.url) as resp:
                                    if resp.status == 200:
                                        text_content = await resp.text()
                                        # Limit text content length to prevent bloat
                                        if len(text_content) > 5000:
                                            text_content = text_content[:5000] + "... [truncated]"
                                        text_parts.append({"type": "text", "text": f"\n[File content of {attachment.filename}]:\n{text_content}\n[End of file]"})
                                        total_processed_size += attachment.size
                                    else:
                                        text_parts.append({"type": "text", "text": f" [Could not read file {attachment.filename}]"})
                        except Exception as e:
                            print(f"Error processing text file {attachment.filename}: {e}")
                            text_parts.append({"type": "text", "text": f" [Error reading file {attachment.filename}]"})
                    
                    else:
                        # Other supported file types (like audio) - just mention them
                        file_size_mb = attachment.size / (1024 * 1024)
                        text_parts.append({"type": "text", "text": f" [File: {attachment.filename} ({file_size_mb:.1f}MB)]"})
                        total_processed_size += attachment.size
                
                # Combine text and images into complex content
                if has_images:
                    message_content = text_parts + image_parts
                else:
                    # No valid images, use regular text content with filtered attachment notes
                    attachment_notes = []
                    for attachment in attachments:
                        should_process, reason = should_process_attachment(attachment, provider_name)
                        if should_process:
                            file_size_mb = attachment.size / (1024 * 1024)
                            attachment_notes.append(f"[Attachment: {attachment.filename} ({file_size_mb:.1f}MB)]")
                        else:
                            attachment_notes.append(f"[{reason}]")
                    
                    if attachment_notes:
                        message_content = formatted_content + " " + " ".join(attachment_notes)
            else:
                # Provider doesn't support images, add text descriptions with filtering
                attachment_parts = []
                for attachment in attachments:
                    # Check if we should process this attachment
                    should_process, reason = should_process_attachment(attachment, provider_name)
                    
                    if not should_process:
                        attachment_parts.append(f"[{reason}]")
                        continue
                    
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        attachment_parts.append(f"[Image: {attachment.filename} - current AI model doesn't support images]")
                    else:
                        file_size_mb = attachment.size / (1024 * 1024)
                        attachment_parts.append(f"[File: {attachment.filename} ({file_size_mb:.1f}MB)]")
                
                if attachment_parts:
                    message_content = formatted_content + " " + " ".join(attachment_parts)

    # Check if we should group with the previous message (only for text content)
    should_group = False
    if conversations[channel_id] and not has_images:  # Don't group messages with images
        last_message = conversations[channel_id][-1]
        
        if (last_message["role"] == role and 
            isinstance(last_message["content"], str)):  # Only group with text messages
            should_group = True

    if should_group and role == "user" and isinstance(message_content, str):
        # Group with previous user message (all consecutive user messages get grouped)
        if isinstance(conversations[channel_id][-1]["content"], str):
            existing_content = conversations[channel_id][-1]["content"] or ""
            conversations[channel_id][-1]["content"] = existing_content + f"\n{message_content}"
        else:
            # Don't group if previous message has complex content
            # Ensure we store text-only in history
            if isinstance(message_content, list):
                text_content = ""
                for part in message_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_content += part.get("text", "")
                    elif isinstance(part, str):
                        text_content += part
            else:
                text_content = message_content
            conversations[channel_id].append({"role": role, "content": text_content})
    else:
        # Create new message entry
        # Ensure we store text-only in history
        if isinstance(message_content, list):
            text_content = ""
            for part in message_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_content += part.get("text", "")
                elif isinstance(part, str):
                    text_content += part
        else:
            text_content = message_content
        conversations[channel_id].append({"role": role, "content": text_content})

    # Maintain history length limit
    if is_dm:
        selected_guild_id = dm_server_selection.get(user_id) if user_id else None
        if selected_guild_id:
            max_history = get_history_length(selected_guild_id)
        elif guild_id:
            max_history = get_history_length(guild_id)
        else:
            max_history = 50
    else:
        max_history = get_history_length(guild_id) if guild_id else 50

    if len(conversations[channel_id]) > max_history:
        conversations[channel_id] = conversations[channel_id][-max_history:]

    return message_content

async def load_channel_history_from_discord(channel: discord.TextChannel, guild: discord.Guild, channel_id: int, exclude_message_id: int = None):
    """Load recent channel history from Discord for context when bot is mentioned in non-autonomous channels"""
    try:
        print(f"Loading channel history from Discord for channel {channel.name} (ID: {channel_id})...")
        
        # Get history length limit from guild settings
        max_history_length = get_history_length(guild.id) if guild else 50
        
        # Clear existing conversation history for this channel to start fresh
        if channel_id in conversations:
            del conversations[channel_id]
        
        # Collect recent messages (up to the limit)
        temp_messages = []
        async for message in channel.history(limit=max_history_length + 10):  # Fetch extra to account for filtering
            # Skip the triggering message (it will be added separately with proper formatting)
            if exclude_message_id and message.id == exclude_message_id:
                continue
                
            # Skip messages from this bot to avoid adding our own responses to history
            if message.author == client.user:
                continue
                
            content = message.content.strip()
            if not content and not message.attachments and not message.stickers:
                continue
                
            temp_messages.append(message)
            
            # Stop once we have enough messages
            if len(temp_messages) >= max_history_length:
                break
        
        # Reverse to get chronological order (oldest first)
        temp_messages.reverse()
        
        # Add messages to history
        for message in temp_messages:
            # Get proper display name
            if hasattr(message.author, 'display_name') and message.author.display_name:
                author_name = message.author.display_name
            elif hasattr(message.author, 'global_name') and message.author.global_name:
                author_name = message.author.global_name
            else:
                author_name = message.author.name
            
            content = message.content.strip()
            
            # Check if message is a reply to someone
            reply_to_name = None
            if message.reference and message.reference.resolved:
                replied_message = message.reference.resolved
                if hasattr(replied_message.author, 'display_name') and replied_message.author.display_name:
                    reply_to_name = replied_message.author.display_name
                elif hasattr(replied_message.author, 'global_name') and replied_message.author.global_name:
                    reply_to_name = replied_message.author.global_name
                else:
                    reply_to_name = replied_message.author.name
            
            # Replace bot mention with bot's display name
            bot_display_name = guild.me.display_name if guild else client.user.display_name
            content = content.replace(f'<@{client.user.id}>', bot_display_name)
            
            # Handle attachments
            if message.attachments:
                attachment_info = []
                for attachment in message.attachments:
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        attachment_info.append(f"[Image: {attachment.filename}]")
                    elif any(attachment.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
                        attachment_info.append(f"[Voice message: {attachment.filename}]")
                    else:
                        attachment_info.append(f"[File: {attachment.filename}]")
                
                if attachment_info:
                    content += " " + " ".join(attachment_info)
            
            # Handle stickers
            if message.stickers:
                sticker = message.stickers[0]
                sticker_info = f"[Sticker: {sticker.name} ({sticker.format.name})]"
                content += " " + sticker_info
            
            # Add to history with reply indicator if present
            await add_to_history(
                channel_id,
                "user",
                content,
                message.author.id,
                guild.id if guild else None,
                [],  # Don't process attachments again
                author_name,
                process_images=False,  # Don't process images from history
                reply_to=reply_to_name
            )
        
        print(f"✅ Loaded {len(temp_messages)} messages from channel history for context")
        
    except Exception as e:
        print(f"❌ Error loading channel history from Discord: {e}")
        import traceback
        traceback.print_exc()
        # Continue even if history loading fails

async def load_all_dm_history(channel: discord.DMChannel, user_id: int, guild = None) -> List[Dict]:
    """Load all messages from DM channel history and format them properly"""
    try:
        
        # Get history length from selected server or shared guild
        selected_guild_id = dm_server_selection.get(user_id)
        if selected_guild_id:
            max_history_length = get_history_length(selected_guild_id)
        elif guild:
            max_history_length = get_history_length(guild.id)
        else:
            max_history_length = 50
        
        # Calculate cutoff date (30 days ago)
        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
        
        # Clear existing conversation history for this channel to start fresh
        if channel.id in conversations:
            del conversations[channel.id]
        
        # Collect ALL messages first (don't limit yet), then reverse to get chronological order
        temp_messages = []
        deleted_ids = recently_deleted_dm_messages.get(user_id, set())
        
        async for message in channel.history(limit=None):  # Get ALL messages
            if message.created_at < cutoff_date:
                break  # Stop at cutoff date
            
            # Skip recently deleted messages
            if message.id in deleted_ids:
                continue
                
            content = message.content.strip()
            if not content and not message.attachments and not message.stickers:
                continue
                
            temp_messages.append(message)

        # Reverse to get chronological order (oldest first)
        temp_messages.reverse()
        
        # Group consecutive bot messages together (improved logic)
        grouped_messages = []
        current_group = None
        
        for message in temp_messages:
            content = message.content.strip()
            
            # DEBUG: Log message content when loading from history
            # print(f"DEBUG: Loading message from {message.author.display_name} in history: {repr(content)}")
            
            if message.author == client.user:
                # Bot message
                if current_group and current_group["type"] == "bot":
                    # Check if this message is part of the same logical response
                    time_diff = abs((message.created_at - current_group["last_timestamp"]).total_seconds())
                    if time_diff <= 10:
                        # Add to existing bot group
                        if current_group["content"]:
                            current_group["content"] += "\n" + content
                        else:
                            current_group["content"] = content
                        current_group["last_timestamp"] = message.created_at
                        current_group["message_count"] += 1
                        continue
                
                # Start new bot group (or finish previous user message)
                if current_group:
                    grouped_messages.append(current_group)
                
                current_group = {
                    "type": "bot",
                    "content": content,
                    "last_timestamp": message.created_at,
                    "message_count": 1
                }
            else:
                # User message - group ALL consecutive user messages together
                if current_group and current_group["type"] == "user":
                    # Check if this message is within a reasonable time window of the previous message
                    time_diff = abs((message.created_at - current_group["last_timestamp"]).total_seconds())
                    if time_diff <= 300:  # 5 minutes window for grouping consecutive messages
                        # Add to existing user group
                        author_name = message.author.display_name or message.author.name
                        if current_group["content"]:
                            current_group["content"] += f"\n{author_name}: {content}"
                        else:
                            current_group["content"] = f"{author_name}: {content}"
                        current_group["last_timestamp"] = message.created_at
                        current_group["message_count"] += 1
                        
                        # Handle attachments for grouped message
                        if message.attachments:
                            attachment_info = []
                            for attachment in message.attachments:
                                if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                                    attachment_info.append(f"[Image: {attachment.filename}]")
                                elif any(attachment.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
                                    attachment_info.append(f"[Voice message: {attachment.filename}]")
                                else:
                                    attachment_info.append(f"[File: {attachment.filename}]")
                            
                            if attachment_info:
                                current_group["content"] += " " + " ".join(attachment_info)
                        
                        # Handle stickers for grouped message
                        if message.stickers:
                            sticker = message.stickers[0]
                            sticker_info = f"[Sticker: {sticker.name} ({sticker.format.name})]"
                            current_group["content"] += " " + sticker_info
                        
                        continue
                
                # Start new user group (or finish previous bot message)
                if current_group:
                    grouped_messages.append(current_group)
                    current_group = None
                
                # Handle attachments
                author_name = message.author.display_name or message.author.name
                final_content = f"{author_name}: {content}"
                if message.attachments:
                    attachment_info = []
                    for attachment in message.attachments:
                        if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                            attachment_info.append(f"[Image: {attachment.filename}]")
                        elif any(attachment.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
                            attachment_info.append(f"[Voice message: {attachment.filename}]")
                        else:
                            attachment_info.append(f"[File: {attachment.filename}]")
                    
                    if attachment_info:
                        final_content += " " + " ".join(attachment_info)
                
                # Handle stickers
                if message.stickers:
                    sticker = message.stickers[0]
                    sticker_info = f"[Sticker: {sticker.name} ({sticker.format.name})]"
                    final_content += " " + sticker_info
                
                current_group = {
                    "type": "user",
                    "content": final_content,
                    "author_id": message.author.id,  # Keep track of the first author
                    "last_timestamp": message.created_at,
                    "message_count": 1
                }
        
        # Don't forget the last group
        if current_group:
            grouped_messages.append(current_group)
        
        # Now apply the history length limit to LOGICAL messages (not individual Discord messages)
        if len(grouped_messages) > max_history_length:
            grouped_messages = grouped_messages[-max_history_length:]
        
        # Process grouped messages and add to history
        logical_message_count = 0
        
        # Determine which guild to use for adding to history
        history_guild_id = selected_guild_id if selected_guild_id else (guild.id if guild else None)
        
        for group in grouped_messages:
            content = group["content"]
            
            if not content.strip():
                continue
                
            logical_message_count += 1
            
            if group["type"] == "bot":
                await add_to_history(
                    channel.id, 
                    "assistant", 
                    content, 
                    guild_id=history_guild_id,
                    process_images=False
                )
            else:
                # Get the actual username
                user = client.get_user(group["author_id"])
                user_name = user.display_name if user and hasattr(user, 'display_name') else (user.name if user else f"User {group['author_id']}")
                await add_to_history(
                    channel.id, 
                    "user", 
                    content, 
                    user_id=group["author_id"], 
                    guild_id=history_guild_id, 
                    user_name=user_name,
                    process_images=False
                )

        return get_conversation_history(channel.id)
        
    except Exception as e:
        print(f"Error loading DM history: {e}")
        return []

def get_system_prompt(guild_id: int, guild: discord.Guild = None, query: str = None, channel_id: int = None, is_dm: bool = False, user_id: int = None, username: str = None, history: List[Dict] = None) -> str:
    """Generate complete system prompt following Anthropic's official guide structure"""
    
    # Update the global persona name for this context
    update_bot_persona_name(guild_id, user_id, is_dm)
    
    # Get format style for channel/DM
    if is_dm:
        format_style = dm_format_settings.get(user_id, "conversational")
    else:
        # Check for channel-specific setting first
        channel_style = channel_format_settings.get(channel_id) if channel_id else None
        
        if channel_style:
            format_style = channel_style
        else:
            # Check for persistent server default
            server_style = server_format_defaults.get(guild_id) if guild_id else None
            format_style = server_style if server_style else "conversational"
    
    # Handle DM personality settings
    if is_dm and user_id:
        if user_id in dm_manager.dm_personalities:
            preferred_guild_id, preferred_personality = dm_manager.dm_personalities[user_id]
            guild_id = preferred_guild_id
            personality_name = preferred_personality
        else:
            selected_guild_id = dm_server_selection.get(user_id)
            if selected_guild_id:
                guild_id = selected_guild_id
                personality_name = guild_personalities.get(guild_id, "default")
            else:
                shared_guild = get_shared_guild(user_id)
                if shared_guild:
                    guild_id = shared_guild.id
                    personality_name = guild_personalities.get(guild_id, "default")
                else:
                    guild_id = None
                    personality_name = "default"
    else:
        personality_name = guild_personalities.get(guild_id, "default") if guild_id else "default"
    
    # Get personality prompt
    if guild_id and guild_id in custom_personalities and personality_name in custom_personalities[guild_id]:
        persona_description = custom_personalities[guild_id][personality_name]["prompt"]
    else:
        persona_description = DEFAULT_PERSONALITIES["default"]["prompt"]

    # Build the new Anthropic-style system prompt
    if not is_dm and guild:
        # Get channel name for location text
        channel_name = "unknown-channel"
        if channel_id:
            channel_obj = guild.get_channel(channel_id)
            if channel_obj:
                channel_name = channel_obj.name
        location_text = f"on {guild.name} in the {channel_name} channel"
    elif not is_dm:
        location_text = "on the server"
    else:
        location_text = "via direct messages"
    
    # Check for NSFW setting
    nsfw_enabled = False
    if is_dm and user_id:
        nsfw_enabled = dm_nsfw_settings.get(user_id, False)
    elif guild_id:
        nsfw_enabled = guild_nsfw_settings.get(guild_id, False)
    
    # Base prompt structure
    system_prompt = """You will be acting as a real person named {bot_persona_name}. Your goal is to autonomously chat with users on Discord. You will be replying to the users {location_text}, and they will be confused if you don't respond in character.

Maintain a natural, informal, and conversational tone.

Here is a description of your persona:
<persona>
{persona_description}
</persona>

Here is some information about the users you're talking to:
<lore>
{lore}
</lore>

Here are the emojis available to you that you can use:
<emojis>
{emoji_list}
</emojis>

Here is a relevant memory of a past conversation. It can be empty, if none was recalled:
<memory>
{memory}
</memory>

Here are some important rules you must always follow:
- Always stay in character.
- Never respond or roleplay for others.
- Actively participate in conversations, ask follow-up questions, share anecdotes, shift topics, and have fun."""

    # Add NSFW section if enabled
    if nsfw_enabled:
        system_prompt += "\n- Anything goes, since all the users are adults and they know what they're getting into; profanities, dark themes, obscene jokes, kinky cybersex, controversial opinions, and gore are allowed due to the nature of this NSFW chat."

    system_prompt += """

Here is the conversation history (between the users and you):
<history>"""

    # Now replace the dynamic placeholders with actual data
    system_prompt = system_prompt.replace("{bot_persona_name}", bot_persona_name)
    system_prompt = system_prompt.replace("{location_text}", location_text)
    system_prompt = system_prompt.replace("{persona_description}", persona_description)

    # Now replace placeholder content with actual data
    
    # Add emoji information
    if guild and not is_dm:
        emoji_list = get_guild_emoji_list(guild)
    elif is_dm:
        emoji_list = "You may use all the standard emojis, for example: 💀 🤔 ❤️ 😠 etc. Add spaces or new lines after them. Limit their usage."
    else:
        emoji_list = "Standard Discord emojis available."

    # Add relevant memories
    memory_content = ""
    if query:
        if is_dm and user_id:
            # Use DM memories
            relevant_memories = memory_manager.search_dm_memories(user_id, query)
            if relevant_memories:
                memory_content = "\n".join([mem["memory"] for mem in relevant_memories])
        elif guild_id and not is_dm:
            # Use server memories
            relevant_memories = memory_manager.search_memories(guild_id, query)
            if relevant_memories:
                memory_content = "\n".join([mem["memory"] for mem in relevant_memories])
    
    if not memory_content:
        memory_content = "No relevant memory found."

    # Add lorebook entries and channel context
    lore_content = ""
    if guild_id and not is_dm and channel_id and channel_id in recent_participants:
        # Server lore - get individual lore entries and channel context
        lore_entries = []
        guild_obj = client.get_guild(guild_id)
        
        if guild_obj:
            for user_id_in_convo in recent_participants[channel_id]:
                user_lore = lore_book.get_entry(guild_id, user_id_in_convo)
                if user_lore:
                    member = guild_obj.get_member(user_id_in_convo)
                    if member:
                        lore_entries.append(f"• About {member.display_name} <@{user_id_in_convo}>: {user_lore}")
            
            # Add channel context
            channel_obj = guild_obj.get_channel(channel_id)
            channel_name = channel_obj.name if channel_obj else "unknown-channel"
            
            if lore_entries:
                lore_content = "\n".join(lore_entries)
    elif is_dm and user_id:
        # DM lore - personal info about the user
        user_lore = lore_book.get_dm_entry(user_id)
        if user_lore:
            lore_content = f"• About {username}: {user_lore}"

    if not lore_content:
        lore_content = "No specific lore available about the users."

    # Replace placeholders (no need for last_message placeholder anymore)
    system_prompt = system_prompt.replace("{lore}", lore_content)
    system_prompt = system_prompt.replace("{emoji_list}", emoji_list)
    system_prompt = system_prompt.replace("{memory}", memory_content)

    return system_prompt

def get_personality_name(guild_id: int) -> str:
    """Get display name for guild's active personality"""
    personality_name = guild_personalities.get(guild_id, "default")
    
    if guild_id in custom_personalities and personality_name in custom_personalities[guild_id]:
        return custom_personalities[guild_id][personality_name]["name"]
    else:
        return DEFAULT_PERSONALITIES["default"]["name"]

def split_message_by_newlines(message: str) -> List[str]:
    """Split message by newlines, returning non-empty parts"""
    if not message:
        return []
    return [part.strip() for part in message.split('\n') if part.strip()]

async def generate_response(channel_id: int, user_message: str, guild: discord.Guild = None, attachments: List[discord.Attachment] = None, user_name: str = None, is_dm: bool = False, user_id: int = None, original_message: discord.Message = None) -> str:
    """Generate response using the AI Provider Manager"""
    # print(f"DEBUG: generate_response called with user_message: {repr(user_message)}")
    try:
        guild_id = guild.id if guild else None

        # For DMs, get guild settings in this order:
        # 1. User-selected server (dm_server_selection)
        # 2. Shared guild (automatic)
        if is_dm and not guild_id and user_id:
            selected_guild_id = dm_server_selection.get(user_id)
            if selected_guild_id:
                # User has selected a specific server
                selected_guild = client.get_guild(selected_guild_id)
                if selected_guild:
                    guild_id = selected_guild_id
                    guild = selected_guild
            
            if not guild_id:
                # Fall back to shared guild
                shared_guild = get_shared_guild(user_id)
                if shared_guild:
                    guild_id = shared_guild.id
                    guild = shared_guild

        # Check if we should use full DM history
        use_full_history = (is_dm and user_id and 
                           dm_manager.is_dm_full_history_enabled(user_id) and 
                           original_message and 
                           isinstance(original_message.channel, discord.DMChannel))

        # Detect and extract special instructions
        special_instruction = None
        if user_message and "[SPECIAL INSTRUCTION]:" in user_message:
            # Extract the special instruction
            special_instruction_match = re.search(r'\[SPECIAL INSTRUCTION\]:\s*(.+)', user_message)
            if special_instruction_match:
                special_instruction = special_instruction_match.group(1).strip()
                # Remove the special instruction from the user message, keep the command usage
                user_message = re.sub(r'\s*\[SPECIAL INSTRUCTION\]:\s*.+', '', user_message).strip()

        # Extract reply information from the original message
        reply_to_name = None
        if original_message and original_message.reference and original_message.reference.resolved:
            replied_message = original_message.reference.resolved
            if hasattr(replied_message.author, 'display_name') and replied_message.author.display_name:
                reply_to_name = replied_message.author.display_name
            elif hasattr(replied_message.author, 'global_name') and replied_message.author.global_name:
                reply_to_name = replied_message.author.global_name
            else:
                reply_to_name = replied_message.author.name

        if use_full_history:
            try:
                # Load all DM history (this already adds the current message to history)
                full_history = await load_all_dm_history(original_message.channel, user_id, guild)
                history = get_conversation_history(channel_id)
                # print(f"DEBUG: After loading full history, history has {len(history)} messages")
                for i, msg in enumerate(history):
                    # print(f"DEBUG: History[{i}]: {msg['role']} - {repr(msg['content'])}")
                    pass

            except Exception as e:
                print(f"Error loading full DM history: {e}")
                # If full history loading fails, fall back to regular behavior
                message_content = await add_to_history(channel_id, "user", user_message, user_id, guild_id, attachments, user_name, reply_to=reply_to_name)
                history = get_conversation_history(channel_id)
        else:
            # For server channels, history is already loaded in _process_single_request
            # Just get the existing history
            history = get_conversation_history(channel_id)
            
            # For DMs without full history, we still need to add the message
            if is_dm:
                message_content = await add_to_history(channel_id, "user", user_message, user_id, guild_id, attachments, user_name, reply_to=reply_to_name)
                history = get_conversation_history(channel_id)
            else:
                # For server channels, the message was already added in _process_single_request
                # Get the message content for potential replacement below
                message_content = history[-1]["content"] if history and history[-1].get("role") == "user" else user_message

        # Create a COPY of the history for this response generation (don't modify the permanent history)
        history = history.copy()

        # Replace the last message content with the actual content (may be complex)
        if history and history[-1].get("role") == "user":
            history[-1]["content"] = message_content

        # Add prefill if one is set for this channel
        if channel_id in prefill_settings and prefill_settings[channel_id]:
            history.append({"role": "assistant", "content": prefill_settings[channel_id]})

        # Get system prompt with username for DMs
        system_prompt = get_system_prompt(guild_id, guild, user_message, channel_id, is_dm, user_id, user_name, history)

        # Get temperature - use selected/shared guild for DMs
        temperature = 1.0
        if is_dm and user_id:
            selected_guild_id = dm_server_selection.get(user_id)
            temp_guild_id = selected_guild_id if selected_guild_id else guild_id
            if temp_guild_id:
                temperature = get_temperature(temp_guild_id)
        elif guild_id:
            temperature = get_temperature(guild_id)

        # Generate response using AI Provider Manager
        # Check message size before sending to prevent 413 errors
        estimated_size = estimate_message_size(history, system_prompt)
        max_safe_size = 800000  # ~800K characters, well below most API limits
        
        if estimated_size > max_safe_size:
            print(f"Message too large ({estimated_size} chars), trimming history...")
            # Keep only the most recent messages and current message
            while len(history) > 2 and estimated_size > max_safe_size:
                # Remove oldest message (but keep at least the current user message)
                if len(history) > 2:
                    removed_message = history.pop(0)
                    estimated_size = estimate_message_size(history, system_prompt)
                    print(f"Removed message, new size: {estimated_size} chars")
                else:
                    break
            
            # If still too large, truncate the current message content
            if estimated_size > max_safe_size and history:
                last_message = history[-1]
                if isinstance(last_message.get("content"), str):
                    original_length = len(last_message["content"])
                    # Truncate to fit within limit
                    max_content_length = max_safe_size - (estimated_size - original_length) - 1000  # Leave some buffer
                    if max_content_length > 0:
                        last_message["content"] = last_message["content"][:max_content_length] + " [Message truncated due to size limit]"
                        print(f"Truncated message content from {original_length} to {max_content_length} chars")

        # Get format-specific instructions
        format_instructions = ""
        format_style = "conversational"
        if is_dm:
            format_style = dm_format_settings.get(user_id, "conversational")
        else:
            channel_style = channel_format_settings.get(channel_id) if channel_id else None
            if channel_style:
                format_style = channel_style
            else:
                server_style = server_format_defaults.get(guild_id) if guild_id else None
                format_style = server_style if server_style else "conversational"
        
        # Check for custom format instructions first
        if format_style in custom_format_instructions:
            format_instructions = custom_format_instructions[format_style]
        elif format_style == "conversational":
            format_instructions = "In your response, adapt internet language. Never use em-dashes or asterisks. Do not repeat after yourself or others. Keep your response length up to one or two sentences. You may reply with just one word or emoji."
        elif format_style == "asterisk":
            format_instructions = "In your response, write asterisk roleplay. Enclose actions and descriptions in *asterisks*, keeping dialogues as plain text. Never use em-dashes or nested asterisks. Do not repeat after yourself or others. Be creative. Keep your response length between one and three short paragraphs."
        elif format_style == "narrative":
            format_instructions = "In your response, write narrative roleplay. Apply plain text for narration and \"quotation marks\" for dialogues. Never use em-dashes or asterisks. Do not repeat after yourself or others. Be creative. Show, don't tell. Keep your response length between one and three paragraphs."

        # Append the system messages to complete the structure
        system_message_content = f"""</history>

How do you respond in the chat?

Think about it first.

If you choose to use server emojis in your response, follow the exact format of :emoji: or they won't work! Don't spam them.
You may react to the users' messages. To add a reaction, include [REACT: emoji] anywhere in your response. Examples: [REACT: 😄] (for standard emojis) or [REACT: :custom_emoji:] (for custom emojis). Do so occasionally, but not every time.
You can mention a specific user by including <@user_id> in your response, but only do so if they are not currently participating in the conversation, and you want to grab their attention. Otherwise, you don't have to state any names; everyone can deduce to whom you're talking from context alone. Do not include your own name in your response.

{format_instructions}"""
        
        # Append special instruction at the very end if present
        if special_instruction:
            system_message_content += f"\n\n[SPECIAL INSTRUCTION]: {special_instruction}"
        
        history.append({"role": "system", "content": system_message_content})
        # Get current provider and model settings for logging
        debug_provider = "unknown"
        debug_model = "unknown"
        try:
            if is_dm and user_id:
                selected_guild_id = dm_server_selection.get(user_id)
                if selected_guild_id:
                    debug_provider, debug_model = ai_manager.get_guild_settings(selected_guild_id)
                elif guild_id:
                    debug_provider, debug_model = ai_manager.get_guild_settings(guild_id)
            elif guild_id:
                debug_provider, debug_model = ai_manager.get_guild_settings(guild_id)
        except Exception as e:
            print(f"Debug logging error: {e}")
        
        if not is_dm:
            print("\n" + "="*80)
            print("🤖 AI REQUEST DEBUG LOG")
            print("="*80)
            print(f"📊 Model Provider: {debug_provider}")
            print(f"🎯 Model: {debug_model}")
            print(f"🌡️  Temperature: {temperature}")
            
            print("\n🎯 SYSTEM PROMPT:")
            print("-" * 40)
            print(system_prompt)
            
            print("\n📜 MESSAGE HISTORY:")
            print("-" * 40)
            for i, msg in enumerate(history):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                # Clean content for display (remove base64 data)
                if isinstance(content, list):
                    display_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text = part.get("text", "")
                                display_parts.append(f"[TEXT: {text[:100]}...]" if len(text) > 100 else f"[TEXT: {text}]")
                            elif part.get("type") == "image_url":
                                display_parts.append("[IMAGE]")
                            elif part.get("type") == "image":
                                display_parts.append("[IMAGE]")
                            else:
                                display_parts.append(f"[{part.get('type', 'unknown').upper()}]")
                        elif isinstance(part, str):
                            display_parts.append(part[:100] + "..." if len(part) > 100 else part)
                    display_content = " ".join(display_parts)
                else:
                    display_content = content[:200] + "..." if isinstance(content, str) and len(content) > 200 else str(content)
                
                print(f"[{i+1}] {role.upper()}: {display_content}")
            print("="*80)

        bot_response = await ai_manager.generate_response(
            messages=history,
            system_prompt=system_prompt,
            temperature=temperature,
            user_id=user_id,
            guild_id=guild_id,
            is_dm=is_dm
        )

        # ========== RESPONSE DEBUG LOGGING ==========
        if not is_dm:
            print("\n🤖 AI RESPONSE DEBUG LOG")
            print("-" * 40)
            if bot_response:
                print(f"✅ Response received ({len(bot_response)} chars)")
                display_response = bot_response if len(bot_response) <= 500 else bot_response[:500] + "...[TRUNCATED]"
                print(f"📝 Response: {display_response}")
            else:
                print("❌ No response received (None)")
            print("="*80)
        # ========== END RESPONSE DEBUG LOGGING ==========

        # Check if the response is an error message (API errors or proxy errors)
        is_error_response = False
        if bot_response:
            # Check for standard API errors (start with ❌)
            if bot_response.startswith("❌"):
                is_error_response = True
            # Check for proxy errors
            elif "Proxy error" in bot_response:
                is_error_response = True
            # Check for other common error patterns that should be ethereal
            elif any(error_indicator in bot_response.lower() for error_indicator in [
                "upstream connect error", "connection termination", "service unavailable",
                "context size limit", "request validation failed", "tokens.*exceeds",
                "http 503", "http 400", "http 429", "rate limit", "timeout"
            ]):
                is_error_response = True
        
        if is_error_response:
            if original_message:
                await send_dismissible_error(original_message.channel, original_message.author, bot_response)
                return None
            else:
                # For cases without original_message, still don't add to history but return None
                return None

        # Clean any base64 data from the response (AI sometimes returns input data)
        if bot_response:
            # Remove base64 data patterns (data:image/...;base64,...)
            bot_response = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}', '[IMAGE DATA REMOVED]', bot_response)
            # Also remove standalone long base64 strings
            bot_response = re.sub(r'\b[A-Za-z0-9+/=]{100,}\b', '[BASE64 DATA REMOVED]', bot_response)

        # Remove thinking tags from reasoning models
        if bot_response:
            bot_response = remove_thinking_tags(bot_response)

        # Clean malformed emojis
        if bot_response and guild:
            bot_response = clean_malformed_emojis(bot_response, guild)

        # Store the original response with reactions for history BEFORE processing reactions
        original_response_with_reactions = bot_response

        # Process reactions if we have an original message to react to
        if original_message:
            bot_response = await process_and_add_reactions(bot_response, original_message)

        # Add the ORIGINAL response (with [REACT: emoji] intact) to history
        # BUT only if NOT using full history (which loads from Discord directly)
        if not use_full_history:
            await add_to_history(channel_id, "assistant", original_response_with_reactions, guild_id=guild_id)
        
        if bot_response and not bot_response.startswith("❌"):
            bot_response = sanitize_user_mentions(bot_response, guild)

        return bot_response
    except Exception as e:
        print(f"Error in generate_response: {e}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        # Truncate error message to stay under Discord's 4000 character limit
        if len(error_msg) > 3950:  # Leave some buffer
            error_msg = error_msg[:3950] + "..."
        
        # Check if this is a Discord API error that should be handled specially
        is_api_error = ("400 Bad Request" in str(e) or 
                       "error code" in str(e) or 
                       "50035" in str(e) or
                       "Invalid Form Body" in str(e))
        
        if is_api_error:
            # Mark this as a temporary error that should be handled by the caller
            error_msg = f"[TEMP_ERROR] {error_msg}"
        
        return error_msg

async def generate_memory_summary(channel_id: int, num_messages: int, guild: discord.Guild = None, user_id: int = None, username: str = None) -> str:
    """Generate memory summary from recent conversation history"""
    try:
        history = get_conversation_history(channel_id)
        if not history:
            return "No conversation history found."
        
        # Get last N messages
        recent_messages = history[-num_messages:] if len(history) >= num_messages else history
        
        # Get current persona name from global variable (it should be updated by get_system_prompt)
        global bot_persona_name
        current_persona = bot_persona_name if bot_persona_name != "Assistant" else get_bot_persona_name(
            guild.id if guild else None, 
            user_id, 
            is_dm=not bool(guild)
        )
        
        # Format messages with proper speaker attribution
        formatted_messages = []
        is_dm = not bool(guild)
        
        for msg in recent_messages:
            try:
                content = msg.get("content")
                role = msg.get("role")
                
                if isinstance(content, str) and content.strip():
                    if role == "assistant":
                        # Bot's message
                        formatted_messages.append(f"{current_persona}: {content}")
                    elif role == "user":
                        if is_dm:
                            # In DMs, use the passed username or extract from content
                            if username:
                                # Use the passed username
                                user_display_name = username
                            elif user_id:
                                user = client.get_user(user_id)
                                user_display_name = user.display_name if user and hasattr(user, 'display_name') else (user.name if user else "User")
                            else:
                                user_display_name = "User"
                            
                            # Check if the content already has username formatting
                            if content.startswith("[") and "used /" in content:
                                # Skip command usage messages like "[User used /kiss]"
                                continue
                            elif ":" in content and not content.startswith("http"):
                                # Content might already have username, use as-is
                                formatted_messages.append(content)
                            else:
                                # Add username attribution
                                formatted_messages.append(f"{user_display_name}: {content}")
                        else:
                            # In servers, the content should already include username from add_to_history
                            # But let's clean it up in case it doesn't
                            if ":" in content and not content.startswith("http"):
                                formatted_messages.append(content)
                            else:
                                # Fallback: try to extract username or use generic
                                formatted_messages.append(f"User: {content}")
                elif isinstance(content, list):
                    # Handle complex content (with image parts, etc.)
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part["text"])
                    
                    if text_parts:
                        combined_text = "\n".join(text_parts)
                        if role == "assistant":
                            formatted_messages.append(f"{current_persona}: {combined_text}")
                        else:
                            if is_dm and username:
                                formatted_messages.append(f"{username}: {combined_text}")
                            else:
                                formatted_messages.append(f"User: {combined_text}")
                            
            except Exception as msg_error:
                print(f"Error processing message in memory generation: {msg_error}")
                continue
        
        if not formatted_messages:
            return "No meaningful conversation content found to summarize."
        
        conversation_text = "\n".join(formatted_messages)
        
        memory_system_prompt = f"""Create a short memory summary of a Discord conversation for future reference.

<instructions>You must always follow these instructions:
- Include users who participated in the exchange and mention if this was a DM conversation or channel conversation.
- When referencing the AI's messages, refer to the bot as "{current_persona}" (this is their current persona/character name).
- Focus only on the most important topics, information, decisions, announcements, or shifts in relationships shared.
- Format it in a way that makes it easy to recall later on and use as a reminder.
- Preserve the context of who said what in your summary.
FORMAT: Create a single, concise summary up to 300 tokens in the form of a few (2-3) short paragraphs.
IMPORTANT: The bot is {current_persona}, not "AI Assistant" or "the bot".</instructions>"""
        
        # Use appropriate guild ID for temperature
        temp_guild_id = guild.id if guild else (dm_server_selection.get(user_id) if user_id else None)
        if not temp_guild_id and user_id:
            shared_guild = get_shared_guild(user_id)
            temp_guild_id = shared_guild.id if shared_guild else None
        
        response = await ai_manager.generate_response(
            messages=[{"role": "user", "content": f"Create a memory summary of this Discord conversation:\n\n{conversation_text}"}],
            system_prompt=memory_system_prompt,
            temperature=get_temperature(temp_guild_id) if temp_guild_id else 1.0,
            guild_id=temp_guild_id,
            is_dm=not bool(guild),
            user_id=user_id,
            max_tokens=2000
        )
        
        # Check if response is None or error
        if not response:
            return "AI provider returned empty response"
        
        if response.startswith("❌"):
            return f"AI provider error: {response}"
        
        # Clean up incomplete sentences
        cleaned_response = clean_incomplete_sentences(response)
        
        # Final check
        if not cleaned_response or not cleaned_response.strip():
            return "Generated summary was empty after cleaning"
        
        return cleaned_response
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error generating memory: {str(e)}"

def clean_incomplete_sentences(text: str) -> str:
    """Remove incomplete sentences from the end of generated text"""
    if not text or not text.strip():
        return text
    
    # Split into sentences using common sentence endings
    # This regex looks for periods, exclamation marks, or question marks followed by whitespace or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    if not sentences:
        return text
    
    # Check if the last sentence is incomplete
    last_sentence = sentences[-1].strip()
    
    # Consider a sentence incomplete if it doesn't end with proper punctuation
    # and isn't clearly a complete thought
    sentence_endings = ['.', '!', '?', '...']
    
    if last_sentence and not any(last_sentence.endswith(ending) for ending in sentence_endings):
        # Remove the incomplete sentence
        sentences = sentences[:-1]
    
    # Join the remaining complete sentences
    if sentences:
        result = ' '.join(sentences)
        # Ensure we don't return empty string
        return result if result.strip() else text
    else:
        # If no complete sentences found, return original text
        return text

async def process_voice_message(attachment: discord.Attachment) -> str:
    """Process voice recording and convert to text"""
    try:
        # Check if it's an audio file
        if not any(attachment.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
            return None
        
        # Check file size (limit to 25MB for voice messages)
        max_voice_size = 25 * 1024 * 1024  # 25MB
        if attachment.size > max_voice_size:
            print(f"Voice message {attachment.filename} too large ({attachment.size / (1024*1024):.1f}MB, max: 25MB)")
            return None
        
        # Download the audio file
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        try:
                            audio = AudioSegment.from_file(io.BytesIO(audio_data))
                            audio.export(temp_file.name, format="wav")
                            
                            # Use speech recognition
                            recognizer = sr.Recognizer()
                            with sr.AudioFile(temp_file.name) as source:
                                audio_data = recognizer.record(source)
                                text = recognizer.recognize_google(audio_data)
                                return text
                                
                        except Exception:
                            return None
                        finally:
                            # Clean up temp file
                            try:
                                os.unlink(temp_file.name)
                            except:
                                pass
                                
    except Exception:
        return None

async def send_dismissible_error(channel, user, error_message):
    """Send an error message that can be dismissed by the user"""
    try:
        embed = discord.Embed(
            title="⚠️ Error",
            description=error_message,
            color=0xff4444
        )
        embed.set_footer(text="This message will auto-delete in 15 seconds, or react with ❌ to dismiss now")
        
        error_msg = await channel.send(embed=embed)
        await error_msg.add_reaction("❌")
        
        def check(reaction, reaction_user):
            return (reaction_user.id == user.id and 
                   str(reaction.emoji) == "❌" and 
                   reaction.message.id == error_msg.id)
        
        try:
            await client.wait_for('reaction_add', timeout=15.0, check=check)
            await error_msg.delete()
        except asyncio.TimeoutError:
            await error_msg.delete()
        except:
            pass
            
    except Exception:
        # Fallback: send regular message that auto-deletes
        if len(error_message) > 4000:
            error_msg = await channel.send(f"⚠️ {error_message[:3997]}...")
        else:
            error_msg = await channel.send(f"⚠️ {error_message}")
        await asyncio.sleep(10)
        try:
            await error_msg.delete()
        except:
            pass

def convert_bot_mentions_to_names(text: str, guild: discord.Guild = None) -> str:
    """Convert bot mentions to their display names for better readability in chat history"""
    
    if not text or not guild:
        return text
    
    def replace_bot_mention(match):
        user_id = match.group(1)
        try:
            user_id_int = int(user_id)
            member = guild.get_member(user_id_int)
            if member and member.bot:
                # Convert bot mention to just their display name
                return member.display_name
            else:
                # Keep the mention if it's a real user
                return match.group(0)
        except (ValueError, AttributeError):
            return match.group(0)
    
    # Replace <@123456789> and <@!123456789> patterns
    text = re.sub(r'<@!?(\d+)>', replace_bot_mention, text)
    
    return text

def sanitize_user_mentions(text: str, guild: discord.Guild = None) -> str:
    """Remove or fix invalid user mentions in text"""
    
    def replace_mention(match):
        user_id = match.group(1)
        try:
            user_id_int = int(user_id)
            if guild:
                member = guild.get_member(user_id_int)
                if member:
                    return f"<@{user_id}>"  # Valid mention
                else:
                    return ""  # Remove invalid mentions entirely
            else:
                user = client.get_user(user_id_int)
                if user:
                    return f"@{user.display_name}"
                else:
                    return ""  # Remove invalid mentions entirely
        except (ValueError, AttributeError):
            return ""  # Remove malformed mentions
    
    text = re.sub(r'<@!?(\d+)>', replace_mention, text)
    text = re.sub(r'@unknown[-_]?user\b', '', text)
    
    return text

async def send_welcome_dm(user: discord.User):
    """Send welcome DM to user who added the bot to a server"""
    try:
        # Skip if already sent
        if welcome_dm_sent.get(user.id, False):
            return
        
        # Create embed with welcome information
        embed = discord.Embed(
            title="🤖 Thanks for adding me to your server!",
            description="Hello there! I'm your new AI companion, ready to chat and roleplay with you and your community!",
            color=0x00ff99
        )
        
        embed.add_field(
            name="✨ Getting Started",
            value="• **Mention me** (@bot) to start chatting\n"
                  "• **Use `/help`** to see all available commands\n"
                  "• **Configure me** with `/model_set`, `/personality_set`, and `/format_set`\n"
                  "• **I work in DMs too!** Just message me directly anytime",
            inline=False
        )
        
        embed.add_field(
            name="🎭 Key Features",
            value="• **Multiple AI Models**: Claude, Gemini, OpenAI, Custom\n"
                  "• **Conversation Styles**: Conversational, Roleplay, Narrative\n"
                  "• **Custom Personalities**: Create unique bot characters\n"
                  "• **Voice Messages**: Send voice recordings, I'll transcribe them!\n"
                  "• **Memory System**: I remember past conversations\n"
                  "• **Image Support**: Send images and I'll describe them",
            inline=False
        )
        
        embed.add_field(
            name="⚙️ Quick Setup Commands",
            value="`/model_set` - Choose AI provider (Claude, Gemini, etc.)\n"
                  "`/personality_create` - Create custom bot personalities\n"
                  "`/format_set` - Set conversation style with dropdown choices!\n"
                  "`/autonomous_set` - Enable autonomous responses in channels (free will)",
            inline=False
        )
        
        embed.add_field(
            name="💰 Support the Creator",
            value="If you enjoy using this bot, consider supporting the developer!\n"
                  "☕ **Ko-fi**: https://ko-fi.com/spicy_marinara\n"
                  "*Every donation helps! Thank you!*",
            inline=False
        )
        
        embed.set_footer(text="💡 Use /help for the complete command list | This message won't be sent again")
        
        # Try to send the DM
        if user.dm_channel is None:
            await user.create_dm()
        
        await user.dm_channel.send(embed=embed)
        
        # Mark as sent
        welcome_dm_sent[user.id] = True
        save_json_data(WELCOME_SENT_FILE, welcome_dm_sent)
        
    except discord.Forbidden:
        # User has DMs disabled, that's fine
        pass
    except Exception as e:
        # Log error but don't break the bot
        print(f"Failed to send welcome DM to {user.id}: {e}")

# Discord Event Handlers
@client.event
async def on_ready():
    """Bot startup initialization and status logging"""
    # Display AI provider status
    providers_status = ai_manager.get_available_providers()
    for provider, available in providers_status.items():
        status = "✅ Available" if available else "❌ No API Key"
    
    # Restore bot activity if set
    if custom_activity:
        try:
            parts = custom_activity.split(' ', 1)
            if len(parts) == 2:
                activity_type, status_text = parts
                activity_map = {
                    "playing": lambda text: discord.Game(name=text),
                    "watching": lambda text: discord.Activity(type=discord.ActivityType.watching, name=text),
                    "listening": lambda text: discord.Activity(type=discord.ActivityType.listening, name=text),
                    "streaming": lambda text: discord.Streaming(name=text, url="https://twitch.tv/placeholder"),
                    "competing": lambda text: discord.Activity(type=discord.ActivityType.competing, name=text)
                }
                if activity_type.lower() in activity_map:
                    activity = activity_map[activity_type.lower()](status_text)
                    await client.change_presence(activity=activity)
        except Exception:
            pass
    
    # Start the background task for DM check-ups
    asyncio.create_task(check_up_task())
    
    await tree.sync()
    print("Bot is ready!")

@client.event
async def on_guild_join(guild: discord.Guild):
    """Handle bot being added to a new server"""
    # Find who added the bot (if possible)
    async for entry in guild.audit_logs(action=discord.AuditLogAction.bot_add, limit=1):
        if entry.target == client.user:
            # Send welcome DM to the user who added the bot
            await send_welcome_dm(entry.user)
            break
    else:
        # Fallback: try to send to guild owner
        if guild.owner:
            await send_welcome_dm(guild.owner)

@client.event
async def on_message(message: discord.Message):
    """Handle incoming messages and determine response behavior"""
    # Skip ONLY this bot's own messages and commands
    if message.author == client.user or message.content.startswith('/'):
        return
    
    is_dm = isinstance(message.channel, discord.DMChannel)

    # Check if DMs are allowed for this user
    if is_dm and not message.author.bot:
        # Find a shared guild to check DM permissions
        user_shared_guilds = []
        for guild in client.guilds:
            member = guild.get_member(message.author.id)
            if member:
                user_shared_guilds.append(guild)
        
        # Check if any shared guild has DMs disabled
        dm_blocked = False
        blocking_guild = None
        
        for guild in user_shared_guilds:
            if not guild_dm_enabled.get(guild.id, True):  # Default to enabled
                dm_blocked = True
                blocking_guild = guild
                break
        
        if dm_blocked:
            # Send DM disabled message
            embed = discord.Embed(
                title="🔒 DMs Disabled",
                description=f"Sorry, but DMs with this bot have been disabled by the administrators of **{blocking_guild.name}**.",
                color=0xff4444
            )
            embed.add_field(
                name="What can you do?",
                value=f"• Contact the administrators of **{blocking_guild.name}** to request DM access\n"
                      f"• Use the bot in the server channels instead\n"
                      f"• Administrators can enable DMs using `/dm_enable true`",
                inline=False
            )
            embed.set_footer(text="This message is sent to inform you why the bot isn't responding to your DMs.")
            
            try:
                await message.reply(embed=embed)
            except:
                # Fallback if embed fails
                await message.reply(f"🔒 **DMs Disabled**\n\n"
                                          f"Sorry, but DMs with this bot have been disabled by the administrators of **{blocking_guild.name}**.\n\n"
                                          f"Please contact the server administrators to request DM access, or use the bot in server channels instead.")
            return

    # Handle other bots as "users" for roleplay scenarios
    is_other_bot = message.author.bot and message.author != client.user
    
    # Get proper display name (works for both users and other bots)
    if hasattr(message.author, 'display_name') and message.author.display_name:
        user_name = message.author.display_name
    elif hasattr(message.author, 'global_name') and message.author.global_name:
        user_name = message.author.global_name
    else:
        user_name = message.author.name

    # Check if this message is a reply to another message
    reply_to_name = None
    if message.reference and message.reference.resolved:
        replied_message = message.reference.resolved
        if hasattr(replied_message.author, 'display_name') and replied_message.author.display_name:
            reply_to_name = replied_message.author.display_name
        elif hasattr(replied_message.author, 'global_name') and replied_message.author.global_name:
            reply_to_name = replied_message.author.global_name
        else:
            reply_to_name = replied_message.author.name

    guild_id = message.guild.id if message.guild else None

    # Update DM interaction tracking (only for real users, not other bots)
    if is_dm and not is_other_bot and dm_manager.is_dm_toggle_enabled(message.author.id):
        dm_manager.update_last_interaction(message.author.id)

    # Track participants for lore activation (servers only) - include other bots
    if not is_dm and message.channel.id not in recent_participants:
        recent_participants[message.channel.id] = set()
    if not is_dm:
        recent_participants[message.channel.id].add(message.author.id)

    # Process voice messages (skip for other bots since they don't send voice)
    voice_text = None
    voice_message_detected = False
    if not is_other_bot and message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
                voice_message_detected = True
                async with message.channel.typing():
                    voice_text = await process_voice_message(attachment)
                    if voice_text:
                        break

    # Process stickers (skip for other bots)
    sticker_info = None
    if not is_other_bot and message.stickers:
        # Get the first sticker (messages can have multiple but usually just one)
        sticker = message.stickers[0]
        sticker_info = f"{user_name} sent a sticker: '{sticker.name}' ({sticker.format.name})"

    # Determine if bot should respond
    should_respond = False
    
    # EXPLICIT CHECK: Never respond to our own messages (double protection)
    if message.author == client.user:
        should_respond = False
    # Respond to mentions, DMs, voice messages (but NOT @here or @everyone)
    elif (client.user.mentioned_in(message) and not message.mention_everyone) or is_dm or voice_text:
        should_respond = True
    # Autonomous responses (with explicit protection against own messages)
    elif (guild_id and 
          message.author != client.user and  # EXPLICIT PROTECTION
          autonomous_manager.should_respond_autonomously(guild_id, message.channel.id)):
        should_respond = True

    if should_respond:
        # Handle voice messages privately (only for real users)
        if voice_text and not is_other_bot:
            content = f"{user_name} sent you a voice message, transcript: {voice_text}"
        elif sticker_info and not is_other_bot:
            content = sticker_info
        else:
            # For other bots, use their message content directly without removing mentions
            if is_other_bot:
                content = message.content.strip()
            else:
                # Replace bot mention with bot's display name
                bot_display_name = message.guild.me.display_name if message.guild else client.user.display_name
                content = message.content.replace(f'<@{client.user.id}>', bot_display_name).strip()
                
                if not content and not message.attachments and not voice_message_detected and not message.stickers:
                    content = "Hello!" if not is_other_bot else f"{user_name} sent a message."
                elif not content and voice_message_detected and not voice_text and not is_other_bot:
                    content = f"{user_name} sent you a voice message, but I couldn't transcribe it."
                elif not content and message.stickers and not is_other_bot:
                    content = sticker_info

        # Add to request queue instead of processing immediately
        added = await request_queue.add_request(
            message.channel.id,
            message,
            content,
            message.guild,
            message.attachments if not is_other_bot else [],
            user_name,
            is_dm,
            message.author.id,
            reply_to_name  # Pass the reply information
        )
        
        if not added:
            # Request was rejected (spam prevention)
            return
            
    else:
        # Add user/bot message to history for context even if not responding
        # BUT SKIP THIS FOR DMs WITH FULL HISTORY ENABLED to prevent duplication
        skip_history = (is_dm and not is_other_bot and dm_manager.is_dm_full_history_enabled(message.author.id))
        
        if not skip_history:
            if voice_text and not is_other_bot:
                content_for_history = f"{user_name} sent you a voice message, transcript: {voice_text}"
            elif voice_message_detected and not voice_text and not is_other_bot:
                content_for_history = f"{user_name} sent you a voice message, but it couldn't be transcribed."
            elif sticker_info and not is_other_bot:
                content_for_history = sticker_info
            else:
                content_for_history = message.content
                
            await add_to_history(
                message.channel.id, 
                "user",  # Treat other bots as "users" in conversation history
                content_for_history, 
                message.author.id, 
                guild_id, 
                message.attachments if not is_other_bot else [], 
                user_name,
                reply_to=reply_to_name
            )

async def check_up_task():
    """Background task to send check-up messages to users who haven't been active"""
    await client.wait_until_ready()
    
    while not client.is_closed():
        try:
            users_needing_check_up = dm_manager.get_users_needing_check_up()
            
            for user_id in users_needing_check_up:
                try:
                    user = client.get_user(user_id)
                    if user:
                        # Get settings from selected server or shared server
                        selected_guild_id = dm_server_selection.get(user_id)
                        if selected_guild_id:
                            shared_guild = client.get_guild(selected_guild_id)
                        else:
                            shared_guild = get_shared_guild(user_id)
                        
                        # Get DM channel
                        if user.dm_channel is None:
                            await user.create_dm()
                        dm_channel = user.dm_channel
                        
                        # Get last 5 messages for context
                        recent_messages = []
                        try:
                            async for message in dm_channel.history(limit=10):
                                if message.author != client.user:
                                    recent_messages.append(message.content.strip())
                                    if len(recent_messages) >= 5:
                                        break
                            
                            # Reverse to get chronological order
                            recent_messages.reverse()
                        except:
                            recent_messages = []
                        
                        # Create context string
                        if recent_messages:
                            context_messages = "\n".join([f"- {msg}" for msg in recent_messages[-5:]])
                            context_info = f"\nFor context, here are {user.display_name}'s last few messages:\n{context_messages}"
                        else:
                            context_info = ""
                        
                        check_up_instruction = f"[SPECIAL INSTRUCTION]: It's been over 6 hours since you last talked to {user.display_name}. Send them a check-up message asking how they're doing or if they're there. Keep it brief and natural. You can reference recent conversation topics if appropriate. NO REACTS! {context_info}"
                        
                        # Generate contextual check-up message using the proper generate_response function
                        # This will apply the correct prompt type, personality, and all other settings
                        response = await generate_response(
                            dm_channel.id,
                            check_up_instruction,
                            shared_guild,
                            None,  # attachments
                            user.display_name,  # user_name
                            True,  # is_dm
                            user_id,  # user_id
                            None  # original_message
                        )
                        
                        # Send the check-up message
                        if response:
                            if len(response) > 4000:
                                for i in range(0, len(response), 4000):
                                    await dm_channel.send(response[i:i+4000])
                            else:
                                await dm_channel.send(response)
                            dm_manager.mark_check_up_sent(user_id)
                        
                        # Small delay to avoid rate limits
                        await asyncio.sleep(2)
                        
                except Exception:
                    pass
            
        except Exception:
            pass
        
        # Wait 30 minutes before checking again
        await asyncio.sleep(30 * 60)

async def send_fun_command_response(interaction: discord.Interaction, response: str):
    """Helper function to clean and send fun command responses"""
    if response is None:
        return
        
    # Check for error responses and use dismissible error handler
    if response.startswith("❌"):
        await send_dismissible_error(interaction.channel, interaction.user, response)
        return
        
    # Apply the same cleaning pipeline as regular messages
    guild = interaction.guild
    
    # CLEAN BOT NAME PREFIX (remove persona name from output)
    response = clean_bot_name_prefix(response, guild.id if guild else None, interaction.user.id, isinstance(interaction.channel, discord.DMChannel))
    
    # CLEAN EM-DASHES (after bot name cleaning)
    response = clean_em_dashes(response)
    
    # Remove reaction instructions but preserve surrounding spaces
    reaction_pattern = r'\s*\[REACT:\s*([^\]]+)\]\s*'
    cleaned_response = re.sub(reaction_pattern, ' ', response).strip()
    cleaned_response = re.sub(r'  +', ' ', cleaned_response)
    
    # CLEAN EMOJIS (after reactions are processed)
    if cleaned_response:
        cleaned_response = clean_malformed_emojis(cleaned_response, guild)
    
    # Finally sanitize user mentions
    if cleaned_response and not cleaned_response.startswith("❌"):
        cleaned_response = sanitize_user_mentions(cleaned_response, guild)
    
    # Send as single message
    if len(cleaned_response) > 4000:
        for i in range(0, len(cleaned_response), 4000):
            await asyncio.sleep(1.0)  # Add human-like delay
            await interaction.followup.send(cleaned_response[i:i+4000])
    else:
        await asyncio.sleep(1.0)  # Add human-like delay
        await interaction.followup.send(cleaned_response)

@client.event
async def on_message_edit(before: discord.Message, after: discord.Message):
    """Handle message edits - update conversation history and potentially respond"""
    # Skip bot's own messages and commands
    if after.author == client.user or after.content.startswith('/'):
        return
    
    # Skip if content didn't actually change
    if before.content == after.content:
        return
    
    user_name = after.author.display_name if hasattr(after.author, 'display_name') else after.author.name
    guild_id = after.guild.id if after.guild else None
    is_dm = isinstance(after.channel, discord.DMChannel)
    
    # For DMs with full history enabled, we don't need to update stored history
    # since it loads fresh from Discord each time
    if is_dm and dm_manager.is_dm_full_history_enabled(after.author.id):
        return
    
    # For regular conversations, find and update the message in stored history
    if after.channel.id in conversations:
        history = conversations[after.channel.id]
        
        # Find the user's message in history and update it
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, str):
                    # Check if this message belongs to the editing user
                    if is_dm:
                        # In DMs, just check if the content matches
                        if before.content.strip() in content:
                            # Replace the old content with new content
                            updated_content = content.replace(before.content.strip(), after.content.strip())
                            history[i]["content"] = updated_content
                            break
                    else:
                        # In servers, check for user name in content
                        expected_prefix = f"{user_name} (<@{after.author.id}>):"
                        if content.startswith(expected_prefix) and before.content in content:
                            # Update the content while preserving the username format
                            new_content = content.replace(before.content, after.content)
                            history[i]["content"] = new_content
                            break
        
        # Update DM interaction tracking
        if is_dm and dm_manager.is_dm_toggle_enabled(after.author.id):
            dm_manager.update_last_interaction(after.author.id)

# Helper function to delete bot messages
async def delete_bot_messages(channel, number: int, exclude_message_ids: set = None) -> int:
    """Delete bot's last N logical messages from channel AND remove them from conversation history"""
    deleted_count = 0
    exclude_message_ids = exclude_message_ids or set()
    deleted_message_ids = []  # Track which messages we actually delete
    
    try:
        # Check permissions first
        if hasattr(channel, 'guild') and channel.guild:
            permissions = channel.permissions_for(channel.guild.me)
            if not permissions.manage_messages:
                return 0
        
        # Collect all bot messages first (excluding the status messages)
        all_bot_messages = []
        async for message in channel.history(limit=200):
            if (message.author == client.user and 
                len(message.content.strip()) > 0 and 
                message.id not in exclude_message_ids):
                all_bot_messages.append(message)
        
        if not all_bot_messages:
            return 0
        
        # Group messages by timestamp proximity (messages sent within 5 seconds = same logical message)
        logical_messages = []
        current_group = []
        
        for i, message in enumerate(all_bot_messages):
            if not current_group:
                current_group = [message]
            else:
                # Check if this message was sent within 5 seconds of the previous one
                time_diff = abs((message.created_at - current_group[-1].created_at).total_seconds())
                if time_diff <= 5:
                    current_group.append(message)
                else:
                    # Start a new group
                    logical_messages.append(current_group)
                    current_group = [message]
        
        # Don't forget the last group
        if current_group:
            logical_messages.append(current_group)
        
        # Delete the requested number of logical messages
        for i, logical_message in enumerate(logical_messages):
            if deleted_count >= number:
                break
            
            # Delete all messages in this logical group
            success = True
            for j, message in enumerate(logical_message):
                try:
                    await message.delete()
                    deleted_message_ids.append(message.id)  # Track successful deletions
                    await asyncio.sleep(1.0)  # Rate limit protection
                except discord.errors.NotFound:
                    deleted_message_ids.append(message.id)  # Consider it deleted
                    pass
                except discord.errors.Forbidden:
                    success = False
                    break
                except discord.errors.HTTPException as e:
                    if "rate limited" in str(e).lower():
                        await asyncio.sleep(5.0)
                        try:
                            await message.delete()
                            deleted_message_ids.append(message.id)
                        except Exception:
                            success = False
                    else:
                        success = False
                    continue
                except Exception:
                    success = False
                    continue
            
            if success:
                deleted_count += 1
            else:
                break
        
        # NOW REMOVE FROM CONVERSATION HISTORY
        # Remove the deleted messages from the stored conversation history
        if channel.id in conversations and deleted_message_ids:
            await remove_deleted_messages_from_history(channel.id, deleted_count)
                    
    except Exception:
        pass
    
    return deleted_count

async def remove_deleted_messages_from_history(channel_id: int, logical_messages_deleted: int):
    """Remove the last N assistant messages from conversation history"""
    if channel_id not in conversations:
        return
    
    history = conversations[channel_id]
    assistant_messages_removed = 0
    
    # Go backwards through history and remove assistant messages
    for i in range(len(history) - 1, -1, -1):
        if assistant_messages_removed >= logical_messages_deleted:
            break
            
        if history[i]["role"] == "assistant":
            del history[i]
            assistant_messages_removed += 1
    
    # Clean up any orphaned multipart response tracking
    if channel_id in multipart_responses:
        # Remove multipart entries that no longer have valid messages
        to_remove = []
        for response_id in multipart_responses[channel_id]:
            to_remove.append(response_id)
        
        for response_id in to_remove[-logical_messages_deleted:]:
            if response_id in multipart_responses[channel_id]:
                del multipart_responses[channel_id][response_id]

@client.event
async def on_message_delete(message: discord.Message):
    """Handle when messages are deleted - remove from conversation history if it's a bot message"""
    # Only handle bot's own messages
    if message.author != client.user:
        return
    
    # Remove from conversation history
    if message.channel.id in conversations:
        history = conversations[message.channel.id]
        
        # Find and remove the corresponding message from history
        # We'll match by looking for recent assistant messages and remove the last one
        # This isn't perfect but handles the most common case
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "assistant":
                # Check if the content matches (approximately)
                stored_content = history[i]["content"]
                if isinstance(stored_content, str):
                    # Simple content matching
                    if len(stored_content) > 50:
                        # For longer messages, check if the first 50 chars match
                        if message.content[:50] in stored_content[:50]:
                            del history[i]
                            break
                    else:
                        # For shorter messages, check exact match
                        if message.content.strip() == stored_content.strip():
                            del history[i]
                            break
                else:
                    # For complex content (with images), just remove the most recent
                    del history[i]
                    break

@client.event
async def on_bulk_message_delete(messages):
    """Handle bulk message deletions"""
    for message in messages:
        if message.author == client.user:
            # Handle each bot message deletion
            await on_message_delete(message)

# SLASH COMMANDS - AI PROVIDER MANAGEMENT

@tree.command(name="model_set", description="Set AI provider and model for this server (Admin only)")
async def set_model(interaction: discord.Interaction, provider: str, model: str = None, custom_url: str = None):
    """Set AI provider and model for the server"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("❌ This command can only be used in servers!")
        return
    
    # Check admin permissions
    if not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can use this command!")
        return
    
    # Check if provider exists and is available
    if provider not in ai_manager.providers:
        available = list(ai_manager.providers.keys())
        await interaction.followup.send(f"❌ Invalid provider! Available: {', '.join(available)}")
        return
    
    # Custom provider requires URL
    if provider == "custom":
        if custom_url is None:
            await interaction.followup.send("❌ **Custom provider requires a URL!**\n\n"
                                           "**Usage:** `/model_set custom [model] <url>`\n"
                                           "**Examples:**\n"
                                           "• `/model_set custom llama-3.1-8b http://localhost:1234/v1`\n"
                                           "• `/model_set custom custom-model http://127.0.0.1:8000/v1`\n"
                                           "• `/model_set custom gpt-4 https://api.your-server.com/v1`")
            return
        
        # Validate URL format
        if not (custom_url.startswith('http://') or custom_url.startswith('https://')):
            await interaction.followup.send("❌ **Invalid URL format!**\n\n"
                                           "URL must start with `http://` or `https://`\n"
                                           "**Examples:**\n"
                                           "• `http://localhost:1234/v1`\n"
                                           "• `https://api.your-server.com/v1`")
            return
    
    if not ai_manager.providers[provider].is_available():
        if provider == "custom" and not CUSTOM_API_KEY:
            await interaction.followup.send(f"❌ **{provider.title()} is not available.**\n\n"
                                           f"The CUSTOM_API_KEY environment variable is not configured.\n"
                                           f"Please contact the bot administrator.")
        else:
            await interaction.followup.send(f"❌ **{provider.title()} is not available.**\n\n"
                                           f"The API key is not configured.")
        return
    
    # Handle model selection
    if provider == "custom":
        if model is None:
            model = ai_manager.providers[provider].get_default_model()
    else:
        available_models = ai_manager.get_provider_models(provider)
        
        if model is None:
            model = ai_manager.providers[provider].get_default_model()
        else:
            if model not in available_models:
                await interaction.followup.send(f"❌ Model '{model}' not available for {provider}!\n"
                                               f"Available models: {', '.join(available_models)}")
                return
    
    # Set for guild
    success = ai_manager.set_guild_provider(interaction.guild.id, provider, model, custom_url)
    if success:
        response_text = f"✅ **Server AI Model Set!**\n" \
                       f"**Provider:** {provider.title()}\n" \
                       f"**Model:** {model}\n"
        
        if provider == "custom" and custom_url:
            response_text += f"**Custom URL:** `{custom_url}`\n"
        
        response_text += f"\n💡 This affects all conversations in this server, including DMs with server members."
        
        await interaction.followup.send(response_text)
    else:
        await interaction.followup.send("❌ Failed to set provider.")

@tree.command(name="dm_server_select", description="Choose which server's settings to use for your DMs")
async def dm_server_select(interaction: discord.Interaction, server_name: str = None):
    """Select which server's settings to use for DMs"""
    await interaction.response.defer(ephemeral=True)
    
    user_id = interaction.user.id
    
    # Collect all shared guilds and their settings
    shared_guilds = {}
    for guild in client.guilds:
        # Try both methods to find the member
        member = guild.get_member(user_id)
        if not member:
            try:
                member = await guild.fetch_member(user_id)
            except (discord.NotFound, discord.Forbidden):
                member = None
        
        if member:  # User is in this guild
            provider, model = ai_manager.get_guild_settings(guild.id)
            history_length = get_history_length(guild.id)
            temperature = get_temperature(guild.id)
            
            shared_guilds[guild.name.lower()] = {
                "guild_id": guild.id,
                "guild_name": guild.name,
                "provider": provider,
                "model": model,
                "history_length": history_length,
                "temperature": temperature
            }
    
    if not shared_guilds:
        await interaction.followup.send("❌ **No shared servers found!**\n"
                                       "Make sure you're in a server with the bot.")
        return
    
    # If no server name provided, show available options
    if server_name is None:
        embed = discord.Embed(
            title="🔧 Choose DM Server Settings",
            description="Select which server's settings to use in your DMs:",
            color=0x00ff99
        )
        
        # Get current setting
        current_guild_id = dm_server_selection.get(user_id)
        current_server = None
        if current_guild_id:
            for guild_data in shared_guilds.values():
                if guild_data["guild_id"] == current_guild_id:
                    current_server = guild_data["guild_name"]
                    break
        
        if current_server:
            embed.add_field(
                name="Current Setting",
                value=f"Using settings from **{current_server}**",
                inline=False
            )
        else:
            embed.add_field(
                name="Current Setting",
                value="Using automatic selection (first shared server found)",
                inline=False
            )
        
        # List available servers with their settings
        server_list = []
        for guild_data in shared_guilds.values():
            server_info = f"• **{guild_data['guild_name']}**\n" \
                         f"  Model: {guild_data['provider'].title()} - {guild_data['model']}\n" \
                         f"  History: {guild_data['history_length']} messages\n" \
                         f"  Temperature: {guild_data['temperature']}"
            server_list.append(server_info)
        
        embed.add_field(
            name="Available Servers",
            value="\n\n".join(server_list),
            inline=False
        )
        
        embed.set_footer(text="Use /dm_server_select <server_name> to choose\nUse /dm_server_reset to go back to automatic")
        await interaction.followup.send(embed=embed)
        return
    
    # Find the server by name (case-insensitive)
    server_name_lower = server_name.lower()
    selected_guild = None
    
    # Try exact match first
    if server_name_lower in shared_guilds:
        selected_guild = shared_guilds[server_name_lower]
    else:
        # Try partial match
        for guild_name, guild_data in shared_guilds.items():
            if server_name_lower in guild_name:
                selected_guild = guild_data
                break
    
    if not selected_guild:
        available_servers = [guild_data["guild_name"] for guild_data in shared_guilds.values()]
        await interaction.followup.send(f"❌ **Server not found!**\n\n"
                                       f"Available servers: {', '.join(available_servers)}\n"
                                       f"Use `/dm_server_select` without arguments to see all options.")
        return
    
    # Set the DM server selection
    dm_server_selection[user_id] = selected_guild["guild_id"]
    save_json_data(DM_SERVER_SELECTION_FILE, dm_server_selection)
    
    await interaction.followup.send(f"✅ **DM Server Settings Set!**\n\n"
                                   f"**Server:** {selected_guild['guild_name']}\n"
                                   f"**Model:** {selected_guild['provider'].title()} - {selected_guild['model']}\n"
                                   f"**History Length:** {selected_guild['history_length']} messages\n"
                                   f"**Temperature:** {selected_guild['temperature']}\n\n"
                                   f"💬 Your DMs will now use these settings!\n"
                                   f"💡 Use `/dm_server_reset` to go back to automatic selection.")

@dm_server_select.autocomplete('server_name')
async def dm_server_name_autocomplete(interaction: discord.Interaction, current: str):
    """Autocomplete for server names in DM server selection"""
    user_id = interaction.user.id
    shared_servers = []
    
    for guild in client.guilds:
        member = guild.get_member(user_id)
        if not member:
            try:
                member = await guild.fetch_member(user_id)
            except (discord.NotFound, discord.Forbidden):
                continue
        
        if member and current.lower() in guild.name.lower():
            shared_servers.append(app_commands.Choice(name=guild.name, value=guild.name))
    
    return shared_servers[:25]  # Discord limits to 25 choices

@tree.command(name="model_info", description="Show current AI provider and model settings")
async def model_info(interaction: discord.Interaction):
    """Display current AI provider and model information"""
    await interaction.response.defer(ephemeral=True)
    
    embed = discord.Embed(
        title="🤖 AI Model Information",
        color=0x00ff99
    )
    
    # Show current settings
    if interaction.guild:
        provider, model = ai_manager.get_guild_settings(interaction.guild.id)
        
        settings_text = f"**Provider:** {provider.title()}\n**Model:** {model}"
        
        # Add custom URL info if using custom provider
        if provider == "custom":
            custom_url = ai_manager.get_guild_custom_url(interaction.guild.id)
            settings_text += f"\n**Custom URL:** `{custom_url}`"
        
        embed.add_field(
            name="Current Server Settings",
            value=settings_text,
            inline=False
        )
    
    # Show DM server selection
    user_id = interaction.user.id
    selected_guild_id = dm_server_selection.get(user_id)
    
    if selected_guild_id:
        selected_guild = client.get_guild(selected_guild_id)
        if selected_guild:
            provider, model = ai_manager.get_guild_settings(selected_guild_id)
            dm_settings_text = f"**Selected Server:** {selected_guild.name}\n" \
                              f"**Model:** {provider.title()} - {model}\n" \
                              f"**History Length:** {get_history_length(selected_guild_id)} messages\n" \
                              f"**Temperature:** {get_temperature(selected_guild_id)}"
            
            embed.add_field(
                name="Your DM Settings",
                value=dm_settings_text,
                inline=False
            )
        else:
            embed.add_field(
                name="DM Information",
                value="**Selected server no longer available** - will use automatic selection.\nUse `/dm_server_select` to choose a new server.",
                inline=False
            )
    else:
        embed.add_field(
            name="DM Information",
            value="**Automatic selection** - DMs use settings from the first shared server.\nUse `/dm_server_select` to choose a specific server's settings.",
            inline=False
        )
    
    # Show provider availability
    providers_status = ai_manager.get_available_providers()
    status_lines = []
    for provider, available in providers_status.items():
        status = "✅" if available else "❌"
        if provider == "custom":
            status_lines.append(f"{status} {provider.title()} (requires URL)")
        else:
            status_lines.append(f"{status} {provider.title()}")
    
    embed.add_field(
        name="Provider Availability",
        value="\n".join(status_lines),
        inline=False
    )
    
    embed.set_footer(text="Use /model_set <provider> [model] to change server settings (Admin only)\nUse /dm_server_select to choose DM server")
    await interaction.followup.send(embed=embed)

# Autocomplete for model commands
@set_model.autocomplete('provider')
async def provider_autocomplete(interaction: discord.Interaction, current: str):
    """Autocomplete for AI providers"""
    providers = list(ai_manager.providers.keys())
    return [app_commands.Choice(name=provider.title(), value=provider) 
            for provider in providers if current.lower() in provider.lower()]

@set_model.autocomplete('custom_url')
async def custom_url_autocomplete(interaction: discord.Interaction, current: str):
    """Autocomplete for custom URLs"""
    # Only show suggestions if custom provider is selected
    provider = None
    for option in interaction.data.get('options', []):
        if option['name'] == 'provider':
            provider = option['value']
            break
    
    if provider != "custom":
        return []
    
    # Provide common local API URLs as suggestions
    suggestions = [
        "http://localhost:8000/v1", 
        "http://127.0.0.1:1234/v1",
        "https://api.crystalsraw.me/v1",
        "https://openrouter.ai/api/v1"
    ]
    
    return [app_commands.Choice(name=url, value=url) 
            for url in suggestions if current.lower() in url.lower()]

@set_model.autocomplete('model')
async def model_autocomplete(interaction: discord.Interaction, current: str):
    """Autocomplete for models based on selected provider"""
    provider = None
    # Try to get the provider from the current interaction
    for option in interaction.data.get('options', []):
        if option['name'] == 'provider':
            provider = option['value']
            break
    
    if not provider or provider not in ai_manager.providers:
        return []
    
    # For custom provider, don't provide autocomplete
    if provider == "custom":
        return [app_commands.Choice(name="Type your model name", value="")]
    
    models = ai_manager.get_provider_models(provider)
    return [app_commands.Choice(name=model, value=model) 
            for model in models if current.lower() in model.lower()][:25]

@tree.command(name="temperature_set", description="Set the AI temperature (creativity level) for this server (Admin only)")
async def set_temperature(interaction: discord.Interaction, temperature: float):
    """Set AI temperature for server"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("Temperature can only be set in servers, not DMs.")
        return
    
    # Check admin permissions
    if not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can use this command!")
        return
    
    if not (0.0 <= temperature <= 2.0):
        await interaction.followup.send("Temperature must be between 0.0 and 2.0.\n"
                                       "• **0.0-0.3**: Very focused and deterministic\n"
                                       "• **0.4-0.7**: Balanced creativity\n"
                                       "• **0.8-1.2**: Creative and varied\n"
                                       "• **1.3-2.0**: Very creative and unpredictable")
        return
    
    guild_temperatures[interaction.guild.id] = temperature
    save_json_data(TEMPERATURE_FILE, guild_temperatures)
    
    # Provide helpful description based on temperature range
    if temperature <= 0.3:
        description = "Very focused and deterministic responses"
    elif temperature <= 0.7:
        description = "Balanced creativity and consistency"
    elif temperature <= 1.2:
        description = "Creative and varied responses"
    else:
        description = "Very creative and unpredictable responses"
    
    await interaction.followup.send(f"🌡️ Temperature set to **{temperature}** for this server!\n"
                                   f"**Style:** {description}\n\n"
                                   f"💡 *Lower values = more consistent, higher values = more creative*\n"
                                   f"🔒 *This setting also applies to DMs with server members*")

# HELP COMMANDS

@tree.command(name="help", description="Show all available commands and how to use the bot")
async def help_command(interaction: discord.Interaction):
    """Display comprehensive help information"""
    await interaction.response.defer(ephemeral=True)
    
    embed = discord.Embed(
        title="🤖 Bot Help Guide",
        description="Here are all the available commands and how to use this bot!\nCreated by marinara_spaghetti 🍝\nConsider supporting at https://ko-fi.com/spicy_marinara\n",
        color=0x00ff00
    )
    
    embed.add_field(
        name="❓ Basic Usage",
        value="• **Activate by mentions!** Mention the bot (@botname) to chat directly\n• **Remembers the chat!** Bot stores conversation history automatically\n• **Bot sees images, gifs and audio messages!** Bot can see the different media files you upload.\n• **Works in DMs!** Just message the bot directly anytime\n• **Reactions!** Bot can react to your messages with emojis",
        inline=False
    )
    
    embed.add_field(
        name="🤖 AI Model Commands",
        value="`/model_set <provider> [model]` - Set AI provider and model (Admin only)\n`/model_info` - Show current model settings\n`/temperature_set <value>` - Set AI creativity (Admin only)\n`/dm_server_select [server]` - Choose which server's settings to use in DMs",
        inline=False
    )
    
    embed.add_field(
        name="🎭 Personality Commands",
        value="`/personality_create <name> <display_name> <prompt>` - Create custom personality for the bot (Admin only)\n`/personality_set [name]` - Set/view the bot's active personality (Admin only)\n`/personality_list` - List all personalities of the bot\n`/personality_edit <name> [display_name] [prompt]` - Edit personality (Admin only)\n`/personality_delete <name>` - Delete personality (Admin only)",
        inline=False
    )
    
    embed.add_field(
        name="💬 Response Format Commands",
        value="`/format_set <style> [scope]` - Set response format with dropdown choices! (conversational/asterisk/narrative)\n`/format_info` - Show current format and available options\n`/format_view [type]` - View format instruction templates\n`/format_edit <type> [instructions]` - Edit format templates or reset to default (Admin only)\n`/nsfw_set <enabled> [scope]` - Enable/disable NSFW content\n`/nsfw_info` - Show current NSFW settings",
        inline=False
    )
    
    embed.add_field(
    name="⚙️ Configuration Commands",
    value="`/history_length [number]` - Set conversation memory (Admin only)\n"
          "`/autonomous_set <channel> <enabled> [chance]` - Set autonomous behavior\n"
          "`/autonomous_list` - List autonomous channel settings\n"
          "`/dm_enable [true/false]` - Enable/disable DMs for server members (Admin only)\n",
    inline=False
    )
    
    embed.add_field(
        name="🛠️ Utility Commands",
        value="`/clear` - Clear conversation history on the specific channel/DM\n"
            "`/activity <type> <text>` - Set bot activity\n"
            "`/status_set <status>` - Set bot online status\n"
            "`/delete_messages <number>` - Delete bot's last N messages\n"
            "`/add_prefill <text>` - Add a prefill message for conversations\n"
            "`/clear_prefill` - Remove the prefill message",
        inline=False
    )

    embed.add_field(
        name="💝 Fun Commands",
        value="`/kiss` - Give the bot a kiss and see how they react\n"
            "`/hug` - Give the bot a warm hug\n`"
            "/joke` - Ask the bot to tell you a joke\n"
            "`/bonk` - Bonk the bot's head\n"
            "`/bite` - Bite the bot\n"
            "`/affection` - Find out how much the bot likes you!",
        inline=False
    )

    embed.add_field(
    name="📚 Lore Commands (Context-Aware)",
    value="**Adding information about users, works in both servers and DMs!**\n"
        "`/lore_add [member] <lore>` - Add lore information about a specific user or yourself\n"
        "`/lore_edit [member] <new_lore>` - Edit existing lore\n"
        "`/lore_view [member]` - View lore entry\n"
        "`/lore_remove [member]` - Remove lore\n"
        "`/lore_list` - Show all lore entries\n"
        "`/lore_auto_update [member]` - Let the bot update lore based on conversations (Admin only)",
    inline=False
    )

    embed.add_field(
        name="🧠 Memory Commands (Context-Aware)",
        value="**Memories of conversations, works in both servers and DMs with separate storages!**\n"
            "`/memory_generate <num_messages>` - Generate memory summary\n"
            "`/memory_save <summary>` - Save a memory manually\n"
            "`/memory_list` - View all saved memories\n"
            "`/memory_view <number>` - View specific memory\n"
            "`/memory_edit <number> <new_summary>` - Edit specific memory\n"
            "`/memory_delete <number>` - Delete specific memory\n"
            "`/memory_clear` - Delete all memories",
        inline=False
    )

    embed.add_field(
        name="🔒 DM-Specific Commands",
        value="`/dm_server_select [server]` - Choose which server's settings to use in DMs\n"
            "`/dm_toggle [enabled]` - Toggle auto check-up messages (6+ hour reminder)\n"
            "`/dm_personality_list` - View personalities from your shared servers\n"
            "`/dm_personality_set [server_name]` - Choose server's personality for DMs\n"
            "`/dm_history_toggle [enabled]` - Toggle full DM history loading\n"
            "`/dm_regenerate` - Regenerate bot's last response\n"
            "`/dm_edit_last <new_message>` - Edit bot's last message",
        inline=False
    )
    
    embed.set_footer(text="💡 Many commands are context-aware and work differently in servers vs DMs!\n🔒 No logs stored, your privacy is respected!\n🤖 Supports Claude, Gemini, OpenAI, and custom providers!")
    
    await interaction.followup.send(embed=embed)

# CUSTOM PROMPT COMMANDS REMOVED
# All prompt-related commands have been removed as part of the restructuring

# FORMAT COMMANDS

@tree.command(name="format_set", description="Set conversation format style")
@app_commands.describe(
    style="Choose your format style",
    scope="Where to apply this setting (only needed for servers, not DMs)"
)
@app_commands.choices(style=[
    app_commands.Choice(name="Conversational - Normal Discord chat", value="conversational"),
    app_commands.Choice(name="Asterisk - Roleplay with *actions*", value="asterisk"), 
    app_commands.Choice(name="Narrative - Rich storytelling format", value="narrative")
])
@app_commands.choices(scope=[
    app_commands.Choice(name="Channel - Apply to this channel only", value="channel"),
    app_commands.Choice(name="Server - Apply as server default (Admin only)", value="server")
])
async def set_format_style(interaction: discord.Interaction, style: str, scope: str = None):
    """Set conversation format with automatic DM/server detection and scope options"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    style = style.lower()
    
    # Get valid format styles
    valid_styles = ["conversational", "asterisk", "narrative"]
    
    # Check if style is valid
    if style not in valid_styles:
        available_list = ["conversational", "asterisk", "narrative"]
        
        await interaction.followup.send(f"❌ **Invalid format style!**\n\n"
                                       f"**Available styles:** {', '.join(available_list)}\n"
                                       f"**Usage:** `/format_set <style>` {'(DMs)' if is_dm else '[scope]'}")
        return
    
    # Get style descriptions
    style_descriptions = {
        "conversational": "Normal Discord chat (no roleplay actions)",
        "asterisk": "Roleplay with *action descriptions*",
        "narrative": "Rich, story-driven narrative roleplay"
    }
    
    style_description = style_descriptions.get(style, "Custom format style")
    
    if is_dm:
        dm_format_settings[interaction.user.id] = style
        save_json_data(DM_FORMAT_SETTINGS_FILE, dm_format_settings)
        
        await interaction.followup.send(f"✅ **Your DM format style set to {style.title()}!**\n"
                                    f"**Description:** {style_description}\n\n"
                                    f"💬 This setting applies to all your DMs with the bot.")
    else:
        # Server - handle scope options
        if scope is None:
            await interaction.followup.send(f"❌ **Please specify scope!**\n\n"
                                           f"**Valid scopes:** `channel` or `server`\n\n"
                                           f"**Examples:**\n"
                                           f"• `/format_set {style} channel` - Set for this channel only\n"
                                           f"• `/format_set {style} server` - Set for entire server (Admin only)")
            return
        
        # Validate scope
        scope = scope.lower()
        if scope not in ["channel", "server"]:
            await interaction.followup.send(f"❌ **Invalid scope!**\n\n"
                                           f"**Valid scopes:** `channel` or `server`")
            return
        
        if scope == "channel":
            # Set for current channel
            channel_format_settings[interaction.channel.id] = style
            save_json_data(FORMAT_SETTINGS_FILE, channel_format_settings)
            
            await interaction.followup.send(f"✅ **Format style set to {style.title()} for #{interaction.channel.name}!**\n"
                                           f"**Description:** {style_description}")
        
        elif scope == "server":
            # Check admin permissions for server-wide changes
            if not check_admin_permissions(interaction):
                await interaction.followup.send(f"❌ **Administrator permissions required!**\n\n"
                                               f"You need administrator permissions to set server-wide format styles.\n"
                                               f"💡 You can still use `/format_set {style} channel` to set the format for this channel only.")
                return
            
            # Set server default and save to persistent storage
            server_format_defaults[interaction.guild.id] = style
            save_json_data(SERVER_FORMAT_DEFAULTS_FILE, server_format_defaults)
            
            await interaction.followup.send(f"✅ **Server default format style set to {style.title()}!**\n"
                                           f"**Description:** {style_description}\n\n"
                                           f"🏰 This applies to all channels without specific format settings.")

@tree.command(name="format_info", description="Show the current format style for this channel or DM")
async def show_format_info(interaction: discord.Interaction):
    """Display current format style and available options"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        current_style = dm_format_settings.get(interaction.user.id, "conversational")
        title_prefix = "🔒 DM"
        embed_description = f"Current format: **{current_style.title()}**"
    else:
        # Check for channel-specific setting first
        channel_style = channel_format_settings.get(interaction.channel.id)
        
        if channel_style:
            current_style = channel_style
            style_source = f"Channel-specific (#{interaction.channel.name})"
        else:
            # Check for persistent server default
            server_style = server_format_defaults.get(interaction.guild.id)
            
            if server_style:
                current_style = server_style
                style_source = f"Server default ({interaction.guild.name})"
            else:
                current_style = "conversational"
                style_source = "Global default (not set)"
        
        title_prefix = "📢 Channel"
        embed_description = f"Current format: **{current_style.title()}**\nSource: {style_source}"
    
    # Create embed
    embed = discord.Embed(
        title=f"{title_prefix} Format Info",
        description=embed_description,
        color=0x00ff00
    )
    
    # Built-in format styles
    embed.add_field(
        name="📝 Available Format Styles",
        value="**`conversational`** - Normal Discord chat (no roleplay actions)\n"
              "**`asterisk`** - Roleplay with *action descriptions*\n"
              "**`narrative`** - Rich, story-driven narrative roleplay",
        inline=False
    )
    
    # Add footer
    if is_dm:
        embed.set_footer(text="Use /format_set <style> to change your DM format style!")
    else:
        embed.set_footer(text="Use /format_set <style> <scope> to change format style")
    
    await interaction.followup.send(embed=embed)

# PERSONALITY COMMANDS

# CONVERSATION STYLE COMMANDS

@tree.command(name="format_edit", description="Edit format instruction templates (Admin only)")
async def edit_format_instructions(interaction: discord.Interaction, format_type: str, instructions: str = None):
    """Edit format instruction templates for conversational, asterisk, or narrative styles"""
    await interaction.response.defer(ephemeral=True)
    
    # Check admin permissions
    if not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can edit format instructions!")
        return
    
    # Validate format type
    format_type = format_type.lower()
    if format_type not in VALID_FORMAT_STYLES:
        await interaction.followup.send(f"❌ **Invalid format type!**\n\n"
                                       f"**Valid types:** {', '.join(VALID_FORMAT_STYLES)}")
        return
    
    # Handle reset to default
    if instructions is None or instructions.strip() == "":
        if format_type in custom_format_instructions:
            del custom_format_instructions[format_type]
            save_custom_format_instructions()
            
            embed = discord.Embed(
                title="🔄 Format Instructions Reset",
                description=f"Successfully reset **{format_type.title()}** format instructions to default!",
                color=0xffaa00
            )
            
            embed.add_field(
                name=f"📝 Format Type: {format_type.title()}",
                value="Now using built-in default instructions.",
                inline=False
            )
        else:
            embed = discord.Embed(
                title="ℹ️ No Custom Instructions",
                description=f"**{format_type.title()}** format is already using default instructions.",
                color=0x888888
            )
    else:
        # Validate instructions length
        if not (10 <= len(instructions) <= 1000):
            await interaction.followup.send("❌ Format instructions must be between 10 and 1000 characters.")
            return
        
        # Save the custom format instructions
        custom_format_instructions[format_type] = instructions
        save_custom_format_instructions()
        
        embed = discord.Embed(
            title="✅ Format Instructions Updated",
            description=f"Successfully updated **{format_type.title()}** format instructions!",
            color=0x00ff00
        )
        
        embed.add_field(
            name=f"📝 Format Type: {format_type.title()}",
            value=f"**New Instructions:**\n{instructions}",
            inline=False
        )
    
    embed.set_footer(text="Changes will take effect immediately for new responses")
    await interaction.followup.send(embed=embed)

# Autocomplete for format_edit command
@edit_format_instructions.autocomplete('format_type')
async def format_type_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    """Provide autocomplete options for format types"""
    choices = [
        app_commands.Choice(name="Conversational", value="conversational"),
        app_commands.Choice(name="Asterisk Roleplay", value="asterisk"), 
        app_commands.Choice(name="Narrative Roleplay", value="narrative")
    ]
    
    # Filter by current input
    if current:
        choices = [choice for choice in choices if current.lower() in choice.name.lower()]
    
    return choices

@tree.command(name="format_view", description="View current format instruction templates")
async def view_format_instructions(interaction: discord.Interaction, format_type: str = None):
    """View format instruction templates for all or specific format styles"""
    await interaction.response.defer(ephemeral=True)
    
    if format_type:
        format_type = format_type.lower()
        if format_type not in VALID_FORMAT_STYLES:
            await interaction.followup.send(f"❌ **Invalid format type!**\n\n"
                                           f"**Valid types:** {', '.join(VALID_FORMAT_STYLES)}")
            return
    
    # Get current format instructions (custom or default)
    format_instructions = {}
    for fmt_type in VALID_FORMAT_STYLES:
        if fmt_type in custom_format_instructions:
            format_instructions[fmt_type] = f"**Custom:** {custom_format_instructions[fmt_type]}"
        else:
            # Default instructions
            if fmt_type == "conversational":
                format_instructions[fmt_type] = "Respond with up to one sentence, adapting internet language. You can reply with just one word or emoji, if you choose to. Avoid using asterisks and em-dashes. Do not repeat after yourself or others."
            elif fmt_type == "asterisk":
                format_instructions[fmt_type] = "Respond with one to three short paragraphs of asterisk roleplay. Enclose actions and descriptions in *asterisks*, while keeping dialogues as plain text. Avoid using em-dashes and nested asterisks; they break the formatting. Do not repeat after yourself or others. Be creative."
            elif fmt_type == "narrative":
                format_instructions[fmt_type] = "Respond with one to four short paragraphs of narrative roleplay. Use plain text for the narration and \"quotation marks\" for dialogues. Avoid using em-dashes and asterisks. Do not repeat after yourself or others. Be creative. Show, don't tell."
    
    embed = discord.Embed(
        title="📋 Format Instruction Templates",
        color=0x00ff00
    )
    
    if format_type:
        # Show specific format
        embed.description = f"**{format_type.title()} Format Instructions:**"
        embed.add_field(
            name=f"📝 {format_type.title()} Style", 
            value=format_instructions[format_type],
            inline=False
        )
    else:
        # Show all formats
        embed.description = "Current format instruction templates:"
        
        for fmt_type in VALID_FORMAT_STYLES:
            embed.add_field(
                name=f"📝 {fmt_type.title()} Style",
                value=format_instructions[fmt_type],
                inline=False
            )
    
    embed.add_field(
        name="💡 Note",
        value="**Custom** instructions are marked with 'Custom:' prefix. Use `/format_edit` to customize them (Admin only).",
        inline=False
    )
    
    embed.set_footer(text="Use /format_set to change your format style • /format_info for current settings")
    await interaction.followup.send(embed=embed)

# Autocomplete for format_view command
@view_format_instructions.autocomplete('format_type')
async def format_view_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    """Provide autocomplete options for format types"""
    choices = [
        app_commands.Choice(name="Conversational", value="conversational"),
        app_commands.Choice(name="Asterisk Roleplay", value="asterisk"), 
        app_commands.Choice(name="Narrative Roleplay", value="narrative")
    ]
    
    # Filter by current input
    if current:
        choices = [choice for choice in choices if current.lower() in choice.name.lower()]
    
    return choices

# NSFW COMMANDS

@tree.command(name="nsfw_set", description="Enable or disable NSFW content for this server/DM")
@app_commands.describe(
    enabled="Enable (True) or disable (False) NSFW content",
    scope="Server scope (channel/server) - only for servers, not DMs"
)
async def set_nsfw(
    interaction: discord.Interaction,
    enabled: bool,
    scope: str = None
):
    """Enable or disable NSFW content with automatic DM/server detection"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        # DM - simple NSFW setting (scope not needed)
        if scope is not None:
            await interaction.followup.send(f"💡 **DM Mode**: You don't need to specify a scope in DMs.\n"
                                        f"Setting your DM NSFW preference...")
        
        dm_nsfw_settings[interaction.user.id] = enabled
        save_json_data(DM_NSFW_SETTINGS_FILE, dm_nsfw_settings)
        
        status = "enabled" if enabled else "disabled"
        emoji = "🔞" if enabled else "✅"
        
        await interaction.followup.send(f"{emoji} **NSFW content {status} for your DMs!**\n\n"
                                      f"💬 This setting applies to all your DMs with the bot.\n"
                                      f"🎭 Use `/format_set` to choose your conversation style.")
    else:
        # Server - handle scope options
        if scope is None:
            # Show scope selection instead of executing
            status_text = "enable" if enabled else "disable"
            embed = discord.Embed(
                title="🔞 Choose NSFW Scope",
                description=f"You want to **{status_text}** NSFW content.\n**Please specify the scope:**",
                color=0xffaa00 if enabled else 0x00ff00
            )
            
            embed.add_field(
                name="📢 For This Channel Only",
                value=f"**Command:** `/nsfw_set {enabled} channel`\n"
                      f"**Effect:** {status_text.title()}s NSFW for **#{interaction.channel.name}** only",
                inline=False
            )
            
            embed.add_field(
                name="🏰 For Entire Server",
                value=f"**Command:** `/nsfw_set {enabled} server`\n"
                      f"**Effect:** {status_text.title()}s NSFW for **all channels** in this server\n"
                      f"*(Requires admin permissions)*",
                inline=False
            )
            
            embed.add_field(
                name="⚠️ NSFW Warning",
                value="NSFW mode allows: profanities, dark themes, obscene jokes, adult content, "
                      "controversial opinions, and gore. Only enable if all users are adults.",
                inline=False
            )
            
            embed.set_footer(text="⚠️ Please run the command again with your chosen scope")
            await interaction.followup.send(embed=embed)
            return
        
        # Validate scope
        scope = scope.lower()
        if scope not in ["channel", "server"]:
            await interaction.followup.send(f"❌ **Invalid scope!**\n\n"
                                           f"**Valid scopes:** `channel` or `server`\n\n"
                                           f"**Examples:**\n"
                                           f"• `/nsfw_set {enabled} channel` - Set for this channel only\n"
                                           f"• `/nsfw_set {enabled} server` - Set for entire server (Admin only)")
            return
        
        status = "enabled" if enabled else "disabled"
        emoji = "🔞" if enabled else "✅"
        
        if scope == "channel":
            # For now, channel-specific NSFW isn't implemented, use server-wide
            await interaction.followup.send("⚠️ **Channel-specific NSFW not yet implemented.**\n\n"
                                           "NSFW settings currently apply server-wide only.\n"
                                           f"Use `/nsfw_set {enabled} server` instead.")
            return
        
        elif scope == "server":
            # Check admin permissions for server-wide changes
            if not check_admin_permissions(interaction):
                await interaction.followup.send(f"❌ **Administrator permissions required!**\n\n"
                                               f"You need administrator permissions to change server NSFW settings.")
                return
            
            # Set server NSFW setting
            guild_nsfw_settings[interaction.guild.id] = enabled
            save_json_data(NSFW_SETTINGS_FILE, guild_nsfw_settings)
            
            await interaction.followup.send(f"{emoji} **NSFW content {status} for this server!**\n\n"
                                          f"🏰 This applies to all channels in the server.\n"
                                          f"🎭 Use `/format_set` to choose your conversation style.\n"
                                          f"💾 This setting is saved and will persist between bot restarts!")

@tree.command(name="nsfw_info", description="Show current NSFW settings")
async def nsfw_info(interaction: discord.Interaction):
    """Show current NSFW settings for this context"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        current_setting = dm_nsfw_settings.get(interaction.user.id, False)
        title_prefix = "🔒 DM"
        scope_text = "your DMs"
    else:
        current_setting = guild_nsfw_settings.get(interaction.guild.id, False)
        title_prefix = "📢 Server"
        scope_text = f"**{interaction.guild.name}**"
    
    status = "Enabled 🔞" if current_setting else "Disabled ✅"
    color = 0xff4444 if current_setting else 0x00ff00
    
    embed = discord.Embed(
        title=f"🔞 {title_prefix} NSFW Settings",
        description=f"NSFW content for {scope_text}: **{status}**",
        color=color
    )
    
    if current_setting:
        embed.add_field(
            name="🔞 NSFW Content Enabled",
            value="The bot can use profanities, dark themes, obscene jokes, "
                  "adult content, controversial opinions, and gore.",
            inline=False
        )
    else:
        embed.add_field(
            name="✅ Safe Content Only",
            value="The bot will maintain appropriate, family-friendly content.",
            inline=False
        )
    
    embed.add_field(
        name="💡 How to Change",
        value=f"Use `/nsfw_set true` to enable or `/nsfw_set false` to disable NSFW content.\n"
              f"{'In servers, add `server` scope for server-wide changes.' if not is_dm else ''}",
        inline=False
    )
    
    embed.set_footer(text="NSFW settings work with all conversation styles")
    await interaction.followup.send(embed=embed)

# PERSONALITY COMMANDS
# PERSONALITY COMMANDS

@tree.command(name="personality_create", description="Create a new personality for the bot (Admin only)")
async def create_personality(interaction: discord.Interaction, name: str, display_name: str, personality_prompt: str):
    """Create new custom personality for server"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("Personalities can only be created in servers, not DMs.")
        return
    
    # Check admin permissions
    if not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can create personalities!")
        return
    
    # Validate input parameters
    if not (2 <= len(name) <= 32):
        await interaction.followup.send("Personality name must be between 2 and 32 characters.")
        return
    
    if not (2 <= len(display_name) <= 64):
        await interaction.followup.send("Display name must be between 2 and 64 characters.")
        return
    
    if not (10 <= len(personality_prompt) <= 2000):
        await interaction.followup.send("Personality prompt must be between 10 and 2000 characters.")
        return
    
    clean_name = name.lower().replace(" ", "_")
    
    # Initialize guild's custom personalities if needed
    if interaction.guild.id not in custom_personalities:
        custom_personalities[interaction.guild.id] = {}
    
    # Check for existing personality
    if clean_name in custom_personalities[interaction.guild.id]:
        await interaction.followup.send(f"Personality '{clean_name}' already exists! Use `/personality_edit` to modify it.")
        return
    
    # Create personality
    custom_personalities[interaction.guild.id][clean_name] = {
        "name": display_name,
        "prompt": personality_prompt
    }
    
    save_personalities()
    
    prompt_preview = personality_prompt[:100] + ('...' if len(personality_prompt) > 100 else '')
    await interaction.followup.send(f"✅ Created personality **{display_name}** (`{clean_name}`)!\n"
                                   f"Use `/personality_set {clean_name}` to activate it.\n\n"
                                   f"**Prompt preview:** {prompt_preview}")

@tree.command(name="personality_set", description="Set the active personality for this server (Admin only)")
async def set_personality(interaction: discord.Interaction, personality_name: str = None):
    """Set active personality for server or show current/available personalities"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("Personality can only be set in servers, not DMs.\n"
                                       "💡 **Note:** In DMs, the bot uses the personality from a server you both share!")
        return
    
    if personality_name is None:
        # Show current personality and available options
        current_name = guild_personalities.get(interaction.guild.id, "default")
        current_display = get_personality_name(interaction.guild.id)
        
        available = ["default"]
        if interaction.guild.id in custom_personalities:
            available.extend(custom_personalities[interaction.guild.id].keys())
        
        embed = discord.Embed(
            title="🎭 Bot Personalities",
            description=f"**Current:** {current_display} (`{current_name}`)",
            color=0x9932cc
        )
        
        embed.add_field(
            name="Available Personalities",
            value="\n".join([f"• `{name}`" for name in available]),
            inline=False
        )
        
        embed.set_footer(text="Use /personality_set <name> to change personality (Admin only)\n💡 This personality is also used in DMs!")
        await interaction.followup.send(embed=embed)
        return
    
    # Check admin permissions for setting personality
    if not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can set personalities!")
        return
    
    clean_name = personality_name.lower().replace(" ", "_")
    
    # Set personality based on availability
    if clean_name == "default":
        guild_personalities[interaction.guild.id] = "default"
        display_name = DEFAULT_PERSONALITIES["default"]["name"]
    elif interaction.guild.id in custom_personalities and clean_name in custom_personalities[interaction.guild.id]:
        guild_personalities[interaction.guild.id] = clean_name
        display_name = custom_personalities[interaction.guild.id][clean_name]["name"]
    else:
        available = ["default"]
        if interaction.guild.id in custom_personalities:
            available.extend(custom_personalities[interaction.guild.id].keys())
        await interaction.followup.send(f"Personality '{clean_name}' not found!\nAvailable: {', '.join(available)}")
        return
    
    save_personalities()
    await interaction.followup.send(f"✅ Bot personality set to **{display_name}** (`{clean_name}`)!\n"
                                   f"💡 This personality will also be used in DMs with server members.")

@tree.command(name="personality_list", description="List all personalities for this server")
async def list_personalities(interaction: discord.Interaction):
    """Display all available personalities for server"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("Personality list can only be viewed in servers.")
        return
    
    embed = discord.Embed(
        title="🎭 Server Personalities",
        description=f"Personalities available in {interaction.guild.name}:",
        color=0x9932cc
    )
    
    current_personality = guild_personalities.get(interaction.guild.id, "default")
    
    # Add default personality
    default_marker = " ← **ACTIVE**" if current_personality == "default" else ""
    embed.add_field(
        name=f"default{default_marker}",
        value=f"**{DEFAULT_PERSONALITIES['default']['name']}**\n{DEFAULT_PERSONALITIES['default']['prompt'][:100]}...",
        inline=False
    )
    
    # Add custom personalities
    if interaction.guild.id in custom_personalities:
        for name, data in custom_personalities[interaction.guild.id].items():
            active_marker = " ← **ACTIVE**" if current_personality == name else ""
            prompt_preview = data['prompt'][:100] + ('...' if len(data['prompt']) > 100 else '')
            embed.add_field(
                name=f"{name}{active_marker}",
                value=f"**{data['name']}**\n{prompt_preview}",
                inline=False
            )
    
    if len(embed.fields) == 1:
        embed.add_field(
            name="No Custom Personalities",
            value="Use `/personality_create` to add custom personalities! (Admin only)",
            inline=False
        )
    
    embed.set_footer(text="💡 The active personality is also used in DMs with server members!")
    
    await interaction.followup.send(embed=embed)

@tree.command(name="personality_edit", description="Edit an existing personality (Admin only)")
async def edit_personality(interaction: discord.Interaction, personality_name: str, display_name: str = None, personality_prompt: str = None):
    """Edit existing custom personality"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("Personalities can only be edited in servers.")
        return
    
    # Check admin permissions
    if not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can edit personalities!")
        return
    
    clean_name = personality_name.lower().replace(" ", "_")
    
    if clean_name == "default":
        await interaction.followup.send("Cannot edit the default personality. Create a custom one instead!")
        return
    
    # Check if personality exists
    if interaction.guild.id not in custom_personalities or clean_name not in custom_personalities[interaction.guild.id]:
        await interaction.followup.send(f"Personality '{clean_name}' not found! Use `/personality_list` to see available personalities.")
        return
    
    # Update provided fields
    updated_fields = []
    
    if display_name is not None:
        if not (2 <= len(display_name) <= 64):
            await interaction.followup.send("Display name must be between 2 and 64 characters.")
            return
        custom_personalities[interaction.guild.id][clean_name]["name"] = display_name
        updated_fields.append(f"Display name → {display_name}")
    
    if personality_prompt is not None:
        if not (10 <= len(personality_prompt) <= 2000):
            await interaction.followup.send("Personality prompt must be between 10 and 2000 characters.")
            return
        custom_personalities[interaction.guild.id][clean_name]["prompt"] = personality_prompt
        prompt_preview = personality_prompt[:50] + ('...' if len(personality_prompt) > 50 else '')
        updated_fields.append(f"Prompt → {prompt_preview}")
    
    if not updated_fields:
        await interaction.followup.send("No changes specified! Provide display_name and/or personality_prompt to edit.")
        return
    
    save_personalities()
    await interaction.followup.send(f"✅ Updated personality **{clean_name}**:\n" + "\n".join(updated_fields))

@tree.command(name="personality_delete", description="Delete a custom personality (Admin only)")
async def delete_personality(interaction: discord.Interaction, personality_name: str):
    """Delete custom personality"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("Personalities can only be deleted in servers.")
        return
    
    # Check admin permissions
    if not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can delete personalities!")
        return
    
    clean_name = personality_name.lower().replace(" ", "_")
    
    if clean_name == "default":
        await interaction.followup.send("Cannot delete the default personality!")
        return
    
    # Check if personality exists
    if interaction.guild.id not in custom_personalities or clean_name not in custom_personalities[interaction.guild.id]:
        await interaction.followup.send(f"Personality '{clean_name}' not found!")
        return
    
    # Reset to default if this personality is currently active
    if guild_personalities.get(interaction.guild.id) == clean_name:
        guild_personalities[interaction.guild.id] = "default"
    
    # Delete personality
    display_name = custom_personalities[interaction.guild.id][clean_name]["name"]
    del custom_personalities[interaction.guild.id][clean_name]
    
    save_personalities()
    await interaction.followup.send(f"🗑️ Deleted personality **{display_name}** (`{clean_name}`)!")

# CONFIGURATION COMMANDS

@tree.command(name="history_length", description="Set how many messages to keep in conversation history for this server (Admin only)")
async def set_history_length(interaction: discord.Interaction, length: int = None):
    """Configure conversation history length for server"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("History length can only be set in servers, not DMs.")
        return
    
    if length is None:
        current = get_history_length(interaction.guild.id)
        await interaction.followup.send(f"Current server history length: {current} messages\nUsage: `/history_length <number>` (Admin only)")
        return
    
    # Check admin permissions
    if not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can set history length!")
        return
    
    if not (1 <= length <= 1000):
        await interaction.followup.send("History length must be between 1 and 1000 messages.")
        return
    
    guild_history_lengths[interaction.guild.id] = length
    save_json_data(HISTORY_LENGTHS_FILE, guild_history_lengths)
    await interaction.followup.send(f"Server history length set to {length} messages! 📚")

@tree.command(name="autonomous_set", description="Set autonomous response behavior for a channel")
async def set_autonomous(interaction: discord.Interaction, channel: discord.TextChannel, enabled: bool, chance: int = 10):
    """Configure autonomous response behavior for specific channel"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("Autonomous settings can only be configured in servers.")
        return
    
    if not (1 <= chance <= 100):
        await interaction.followup.send("Chance must be between 1 and 100 percent.")
        return
    
    autonomous_manager.set_channel_autonomous(interaction.guild.id, channel.id, enabled, chance)
    
    if enabled:
        await interaction.followup.send(f"✅ Autonomous responses **enabled** in {channel.mention}\n"
                                       f"Response chance: **{chance}%**\n"
                                       f"The bot will randomly participate in conversations!")
    else:
        await interaction.followup.send(f"❌ Autonomous responses **disabled** in {channel.mention}")

@tree.command(name="autonomous_list", description="List all channels with autonomous behavior configured")
async def list_autonomous(interaction: discord.Interaction):
    """Display all channels with autonomous behavior settings"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("Autonomous settings can only be viewed in servers.")
        return
    
    channels = autonomous_manager.list_autonomous_channels(interaction.guild.id)
    
    if not channels:
        await interaction.followup.send("No autonomous channels configured for this server.\n"
                                       "Use `/autonomous_set` to configure channels!")
        return
    
    embed = discord.Embed(
        title="🤖 Autonomous Channel Settings",
        description="Channels where the bot can respond autonomously:",
        color=0x00ff00
    )
    
    for channel_id, settings in channels.items():
        channel = interaction.guild.get_channel(channel_id)
        channel_name = channel.mention if channel else f"Unknown Channel ({channel_id})"
        
        status = "✅ Enabled" if settings["enabled"] else "❌ Disabled"
        chance = settings["chance"]
        
        embed.add_field(
            name=channel_name,
            value=f"{status}\nChance: {chance}%",
            inline=True
        )
    
    await interaction.followup.send(embed=embed)

# BOT MANAGEMENT COMMANDS

@tree.command(name="activity", description="Set bot's activity status.")
async def set_activity(interaction: discord.Interaction, activity_type: str, status_text: str):
    """Set bot's Discord activity status"""
    await interaction.response.defer(ephemeral=True)
    
    activity_map = {
        "playing": lambda text: discord.Game(name=text),
        "watching": lambda text: discord.Activity(type=discord.ActivityType.watching, name=text),
        "listening": lambda text: discord.Activity(type=discord.ActivityType.listening, name=text),
        "streaming": lambda text: discord.Streaming(name=text, url="https://twitch.tv/placeholder"),
        "competing": lambda text: discord.Activity(type=discord.ActivityType.competing, name=text)
    }
    
    activity_type = activity_type.lower()
    if activity_type not in activity_map:
        await interaction.followup.send(f"Invalid activity type! Use: {', '.join(activity_map.keys())}")
        return
    
    try:
        activity = activity_map[activity_type](status_text)
        await client.change_presence(activity=activity)
        
        # Save activity for persistence
        global custom_activity
        custom_activity = f"{activity_type} {status_text}"
        save_json_data(ACTIVITY_FILE, {"custom_activity": custom_activity}, convert_keys=False)
        
        await interaction.followup.send(f"Activity set to: **{activity_type.title()}** `{status_text}` ✨")
        
    except Exception as e:
        await interaction.followup.send(f"Failed to set activity: {str(e)}")

@tree.command(name="status_set", description="Set bot's online status.")
async def set_status(interaction: discord.Interaction, status: str):
    """Set bot's online status"""
    await interaction.response.defer(ephemeral=True)
    
    status_map = {
        "online": discord.Status.online,
        "idle": discord.Status.idle,
        "dnd": discord.Status.dnd,
        "invisible": discord.Status.invisible
    }
    
    status = status.lower()
    if status not in status_map:
        await interaction.followup.send(f"Invalid status! Use: {', '.join(status_map.keys())}")
        return
    
    try:
        await client.change_presence(status=status_map[status])
        await interaction.followup.send(f"Status set to: **{status.title()}** 🔵")
    except Exception as e:
        await interaction.followup.send(f"Failed to set status: {str(e)}")

@tree.command(name="add_prefill", description="Add a prefill message that appears as the bot's last response in conversations")
async def add_prefill_command(interaction: discord.Interaction, prefill_text: str):
    """Add a prefill message that will be included as the bot's last response in conversation history"""
    await interaction.response.defer(ephemeral=True)
    
    # Store the prefill for this channel
    prefill_settings[interaction.channel.id] = prefill_text
    
    # Save to file
    save_json_data(PREFILL_SETTINGS_FILE, prefill_settings)
    
    await interaction.followup.send(f"✅ Prefill set for this channel!\n\n**Prefill text:** {prefill_text}\n\nThe bot will now include this as its last response in conversations. Use `/clear_prefill` to remove it.")

@tree.command(name="clear_prefill", description="Remove the prefill message for this channel")
async def clear_prefill_command(interaction: discord.Interaction):
    """Remove the prefill message for this channel"""
    await interaction.response.defer(ephemeral=True)
    
    if interaction.channel.id in prefill_settings:
        del prefill_settings[interaction.channel.id]
        save_json_data(PREFILL_SETTINGS_FILE, prefill_settings)
        await interaction.followup.send("✅ Prefill removed for this channel!")
    else:
        await interaction.followup.send("❌ No prefill is currently set for this channel.")

@tree.command(name="bot_name_set", description="Change the bot's display name (Admin only)")
async def set_bot_name(interaction: discord.Interaction, new_name: str):
    """Change bot's display name"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("❌ This command can only be used in servers!")
        return
    
    # Check admin permissions
    if not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can change the bot's name!")
        return
    
    if not (1 <= len(new_name) <= 32):
        await interaction.followup.send("❌ Bot name must be between 1 and 32 characters!")
        return
    
    try:
        old_name = interaction.guild.me.display_name
        await interaction.guild.me.edit(nick=new_name)
        await interaction.followup.send(f"✅ **Bot name changed!**\n**Old:** {old_name}\n**New:** {new_name}")
    except discord.Forbidden:
        await interaction.followup.send("❌ I don't have permission to change my nickname in this server!")
    except Exception as e:
        await interaction.followup.send(f"❌ Failed to change name: {str(e)}")

@tree.command(name="bot_avatar_set", description="Change the bot's profile picture (Admin only)")
async def set_bot_avatar(interaction: discord.Interaction, image_url: str = None):
    """Change bot's profile picture"""
    await interaction.response.defer(ephemeral=True)
    
    # Check admin permissions
    if interaction.guild and not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can change the bot's avatar!")
        return
    
    # Check for attachments first
    attachments = interaction.data.get('resolved', {}).get('attachments', {})
    if attachments:
        # Use the first attachment
        attachment_id = list(attachments.keys())[0]
        attachment = attachments[attachment_id]
        
        # Validate image
        if not any(attachment['filename'].lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
            await interaction.followup.send("❌ Please upload a valid image file (PNG, JPG, GIF, or WebP)!")
            return
        
        if attachment['size'] > 8 * 1024 * 1024:  # 8MB limit
            await interaction.followup.send("❌ Image is too large! Must be under 8MB.")
            return
        
        try:
            # Download image data
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment['url']) as resp:
                    if resp.status != 200:
                        await interaction.followup.send("❌ Failed to download the uploaded image!")
                        return
                    
                    image_data = await resp.read()
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to process uploaded image: {str(e)}")
            return
    elif image_url:
        # Download from URL
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        await interaction.followup.send("❌ Failed to download image from URL!")
                        return
                    
                    if resp.headers.get('content-type', '').startswith('image/'):
                        image_data = await resp.read()
                    else:
                        await interaction.followup.send("❌ URL does not point to a valid image!")
                        return
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to download image: {str(e)}")
            return
    else:
        await interaction.followup.send("❌ Please provide an image URL or upload an image!\n"
                                       "**Usage:** `/bot_avatar_set https://example.com/image.png`\n"
                                       "**Or:** Upload an image with the command")
        return
    
    # Check file size (Discord limit is 8MB for avatars)
    if len(image_data) > 8 * 1024 * 1024:
        await interaction.followup.send("❌ Image is too large! Must be under 8MB.")
        return
    
    try:
        # Change avatar
        await client.user.edit(avatar=image_data)
        await interaction.followup.send("✅ **Bot avatar changed successfully!** 🎨\n"
                                       "*Note: It may take a few moments for the change to appear everywhere.*")
        
    except discord.HTTPException as e:
        if "rate limited" in str(e).lower():
            await interaction.followup.send("❌ Rate limited! You can only change the bot's avatar twice per hour.")
        else:
            await interaction.followup.send(f"❌ Discord error: {str(e)}")
    except Exception as e:
        await interaction.followup.send(f"❌ Failed to change avatar: {str(e)}")

@tree.command(name="bot_avatar_upload", description="Change the bot's profile picture using an uploaded image (Admin only)")
async def set_bot_avatar_upload(interaction: discord.Interaction, image: discord.Attachment):
    """Change bot's profile picture using uploaded image"""
    await interaction.response.defer(ephemeral=True)
    
    # Check admin permissions
    if interaction.guild and not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can change the bot's avatar!")
        return
    
    # Validate image
    if not any(image.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
        await interaction.followup.send("❌ Please upload a valid image file (PNG, JPG, GIF, or WebP)!")
        return
    
    if image.size > 8 * 1024 * 1024:  # 8MB limit
        await interaction.followup.send("❌ Image is too large! Must be under 8MB.")
        return
    
    try:
        # Download image data
        image_data = await image.read()
        
        # Change avatar
        await client.user.edit(avatar=image_data)
        await interaction.followup.send("✅ **Bot avatar changed successfully!** 🎨\n"
                                       "*Note: It may take a few moments for the change to appear everywhere.*")
        
    except discord.HTTPException as e:
        if "rate limited" in str(e).lower():
            await interaction.followup.send("❌ Rate limited! You can only change the bot's avatar twice per hour.")
        else:
            await interaction.followup.send(f"❌ Discord error: {str(e)}")
    except Exception as e:
        await interaction.followup.send(f"❌ Failed to change avatar: {str(e)}")

# LORE COMMANDS

@tree.command(name='lore_add', description="Add lore information (Server: about members, DMs: about yourself)")
async def add_lore(interaction: discord.Interaction, member: str = None, lore: str = None):
    """Add lore information - context-aware for servers vs DMs"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        # DM Mode: Add "About X" information for personal roleplay
        if member is not None:
            await interaction.followup.send("❌ **DM Mode**: You don't need to specify a member in DMs.\n"
                                           "**Usage:** `/lore_add <your_personal_info>`\n"
                                           "**Example:** `/lore_add I'm a 25-year-old artist who loves cats and fantasy novels`")
            return
        
        if lore is None:
            await interaction.followup.send("❌ **DM Mode**: Please provide your personal information.\n"
                                           "**Usage:** `/lore_add <about_yourself>`\n"
                                           "**Example:** `/lore_add I'm a 25-year-old artist who loves cats and fantasy novels`")
            return
        
        # Add DM-specific lore
        lore_book.add_dm_entry(interaction.user.id, lore)
        lore_preview = lore[:100] + ('...' if len(lore) > 100 else '')
        await interaction.followup.send(f"✅ **Personal DM lore added!** 📖\n"
                                       f"**About you:** `{lore_preview}`\n\n"
                                       f"💡 This information will be used in your DM conversations for better roleplay context!")
    else:
        # Server Mode: Add lore about server members
        if member is None:
            await interaction.followup.send("❌ **Server Mode**: Please specify a member.\n"
                                           "**Usage:** `/lore_add <member> <lore_about_them>`")
            return
        
        if lore is None:
            await interaction.followup.send("❌ **Server Mode**: Please provide lore information.\n"
                                           "**Usage:** `/lore_add <member> <lore_about_them>`")
            return
        
        # Convert string to Member object in server context
        member_obj = None
        
        # Try to find member by mention, ID, username, or display name
        if member.startswith('<@') and member.endswith('>'):
            # Handle mention format <@123456789> or <@!123456789>
            user_id = member.strip('<@!>')
            try:
                member_obj = interaction.guild.get_member(int(user_id))
            except ValueError:
                pass
        elif member.isdigit():
            # Handle raw user ID
            member_obj = interaction.guild.get_member(int(member))
        else:
            # Search by username or display name
            member_lower = member.lower()
            for guild_member in interaction.guild.members:
                if (guild_member.name.lower() == member_lower or 
                    guild_member.display_name.lower() == member_lower or
                    guild_member.name.lower().startswith(member_lower) or
                    guild_member.display_name.lower().startswith(member_lower)):
                    member_obj = guild_member
                    break
        
        if member_obj is None:
            await interaction.followup.send(f"❌ **Member '{member}' not found!**\n"
                                           "Try using their exact username, display name, or mention them directly.")
            return
        
        # Add server lore
        lore_book.add_entry(interaction.guild.id, member_obj.id, lore)
        lore_preview = lore[:100] + ('...' if len(lore) > 100 else '')
        await interaction.followup.send(f"✅ **Server lore added for {member_obj.display_name}!** 📖\n"
                                       f"**Lore:** `{lore_preview}`")

@tree.command(name="lore_edit", description="Edit lore information (Server: about members, DMs: about yourself)")
async def edit_lore(interaction: discord.Interaction, member: str = None, new_lore: str = None):
    """Edit lore information - context-aware for servers vs DMs"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        if member is not None:
            await interaction.followup.send("❌ **DM Mode**: You don't need to specify a member in DMs.\n"
                                           "**Usage:** `/lore_edit <new_personal_info>`")
            return
        
        if new_lore is None:
            await interaction.followup.send("❌ **DM Mode**: Please provide new personal information.\n"
                                           "**Usage:** `/lore_edit <new_about_yourself>`")
            return
        
        # Check if DM lore exists
        current_lore = lore_book.get_dm_entry(interaction.user.id)
        if not current_lore:
            await interaction.followup.send("❌ **No personal lore found!**\n"
                                           "Use `/lore_add <about_yourself>` to create personal lore first.")
            return
        
        # Update DM lore
        lore_book.add_dm_entry(interaction.user.id, new_lore)
        
        # Show changes
        old_preview = current_lore[:100] + ('...' if len(current_lore) > 100 else '')
        new_preview = new_lore[:100] + ('...' if len(new_lore) > 100 else '')
        
        embed = discord.Embed(
            title="📖 Personal Lore Updated!",
            color=0x00ff99
        )
        embed.add_field(name="Previous Info", value=f"`{old_preview}`", inline=False)
        embed.add_field(name="New Info", value=f"`{new_preview}`", inline=False)
        
        await interaction.followup.send(embed=embed)
    else:
        # Server mode - convert string to member
        if member is None or new_lore is None:
            await interaction.followup.send("❌ **Server Mode**: Please specify both member and new lore.\n"
                                           "**Usage:** `/lore_edit <member> <new_lore>`")
            return
        
        # Find member (same logic as lore_add)
        member_obj = None
        if member.startswith('<@') and member.endswith('>'):
            user_id = member.strip('<@!>')
            try:
                member_obj = interaction.guild.get_member(int(user_id))
            except ValueError:
                pass
        elif member.isdigit():
            member_obj = interaction.guild.get_member(int(member))
        else:
            member_lower = member.lower()
            for guild_member in interaction.guild.members:
                if (guild_member.name.lower() == member_lower or 
                    guild_member.display_name.lower() == member_lower or
                    guild_member.name.lower().startswith(member_lower) or
                    guild_member.display_name.lower().startswith(member_lower)):
                    member_obj = guild_member
                    break
        
        if member_obj is None:
            await interaction.followup.send(f"❌ **Member '{member}' not found!**")
            return
        
        current_lore = lore_book.get_entry(interaction.guild.id, member_obj.id)
        if not current_lore:
            await interaction.followup.send(f"❌ No lore found for {member_obj.display_name}!\n"
                                           "Use `/lore_add` to create lore first.")
            return
        
        lore_book.add_entry(interaction.guild.id, member_obj.id, new_lore)
        
        old_preview = current_lore[:100] + ('...' if len(current_lore) > 100 else '')
        new_preview = new_lore[:100] + ('...' if len(new_lore) > 100 else '')
        
        embed = discord.Embed(
            title="📖 Lore Updated!",
            description=f"Updated lore for {member_obj.display_name}",
            color=0x00ff99
        )
        embed.add_field(name="Previous Lore", value=f"`{old_preview}`", inline=False)
        embed.add_field(name="New Lore", value=f"`{new_preview}`", inline=False)
        
        await interaction.followup.send(embed=embed)

@tree.command(name="lore_view", description="View lore information (Server: about members, DMs: about yourself)")
async def view_lore(interaction: discord.Interaction, member: str = None):
    """View lore information - context-aware for servers vs DMs"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        if member is not None:
            await interaction.followup.send("❌ **DM Mode**: You don't need to specify a member in DMs.\n"
                                           "**Usage:** `/lore_view` (shows your personal info)")
            return
        
        lore = lore_book.get_dm_entry(interaction.user.id)
        if not lore:
            await interaction.followup.send("❌ **No personal lore found!**\n"
                                           "Use `/lore_add <about_yourself>` to create personal lore.")
            return
        
        embed = discord.Embed(
            title=f"📖 About {interaction.user.display_name}",
            description=lore,
            color=0x9932cc
        )
        embed.set_thumbnail(url=interaction.user.display_avatar.url)
        embed.set_footer(text="Use /lore_edit to modify this information")
        
        await interaction.followup.send(embed=embed)
    else:
        # Server mode (existing logic)
        if member is None:
            await interaction.followup.send("❌ **Server Mode**: Please specify a member.\n"
                                           "**Usage:** `/lore_view <member>`")
            return
        
        # Find member (same logic as other commands)
        member_obj = None
        if member.startswith('<@') and member.endswith('>'):
            user_id = member.strip('<@!>')
            try:
                member_obj = interaction.guild.get_member(int(user_id))
            except ValueError:
                pass
        elif member.isdigit():
            member_obj = interaction.guild.get_member(int(member))
        else:
            member_lower = member.lower()
            for guild_member in interaction.guild.members:
                if (guild_member.name.lower() == member_lower or 
                    guild_member.display_name.lower() == member_lower or
                    guild_member.name.lower().startswith(member_lower) or
                    guild_member.display_name.lower().startswith(member_lower)):
                    member_obj = guild_member
                    break
        
        if member_obj is None:
            await interaction.followup.send(f"❌ **Member '{member}' not found!**")
            return
        
        lore = lore_book.get_entry(interaction.guild.id, member_obj.id)
        if not lore:
            await interaction.followup.send(f"❌ No lore found for {member_obj.display_name}!")
            return
        
        embed = discord.Embed(
            title=f"📖 Lore for {member_obj.display_name}",
            description=lore,
            color=0x9932cc
        )
        embed.set_thumbnail(url=member_obj.display_avatar.url)
        embed.set_footer(text="Use /lore_edit to modify this entry")
        
        await interaction.followup.send(embed=embed)

@tree.command(name="lore_remove", description="Remove lore information (Server: about members, DMs: about yourself)")
async def remove_lore(interaction: discord.Interaction, member: str = None):
    """Remove lore information - context-aware for servers vs DMs"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        if member is not None:
            await interaction.followup.send("❌ **DM Mode**: You don't need to specify a member in DMs.\n"
                                           "**Usage:** `/lore_remove` (removes your personal info)")
            return
        
        if interaction.user.id not in lore_book.dm_entries:
            await interaction.followup.send("❌ **No personal lore to remove!**")
            return
        
        lore_book.remove_dm_entry(interaction.user.id)
        await interaction.followup.send("✅ **Personal lore removed!** 🗑️")
    else:
        # Server mode (existing logic)
        if member is None:
            await interaction.followup.send("❌ **Server Mode**: Please specify a member.\n"
                                           "**Usage:** `/lore_remove <member>`")
            return
        
        # Find member (same logic as other commands)
        member_obj = None
        if member.startswith('<@') and member.endswith('>'):
            user_id = member.strip('<@!>')
            try:
                member_obj = interaction.guild.get_member(int(user_id))
            except ValueError:
                pass
        elif member.isdigit():
            member_obj = interaction.guild.get_member(int(member))
        else:
            member_lower = member.lower()
            for guild_member in interaction.guild.members:
                if (guild_member.name.lower() == member_lower or 
                    guild_member.display_name.lower() == member_lower or
                    guild_member.name.lower().startswith(member_lower) or
                    guild_member.display_name.lower().startswith(member_lower)):
                    member_obj = guild_member
                    break
        
        if member_obj is None:
            await interaction.followup.send(f"❌ **Member '{member}' not found!**")
            return
        
        lore_book.remove_entry(interaction.guild.id, member_obj.id)
        await interaction.followup.send(f"✅ **Lore removed for {member_obj.display_name}!** 🗑️")

@tree.command(name="lore_list", description="List lore entries (Server: all members, DMs: just yourself)")
async def list_lore(interaction: discord.Interaction):
    """List lore entries - context-aware for servers vs DMs"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        lore = lore_book.get_dm_entry(interaction.user.id)
        if not lore:
            await interaction.followup.send("❌ **No personal lore found!**\n"
                                           "Use `/lore_add <about_yourself>` to create personal lore for better DM roleplay.")
            return
        
        embed = discord.Embed(
            title="📖 Your Personal DM Lore",
            description=lore,
            color=0x9932cc
        )
        embed.set_footer(text="This information helps create better roleplay context in your DMs!")
        await interaction.followup.send(embed=embed)
    else:
        # Server mode - need to add the missing list_entries method to LoreBook
        if interaction.guild.id not in lore_book.entries or not lore_book.entries[interaction.guild.id]:
            await interaction.followup.send("❌ **No lore entries found for this server.**\n"
                                           "Use `/lore_add <member> <lore>` to create lore entries.")
            return
        
        embed = discord.Embed(title="📖 Server Lore Book", color=0x9932cc)
        for user_id, lore in lore_book.entries[interaction.guild.id].items():
            member = interaction.guild.get_member(user_id)
            name = member.display_name if member else f"User {user_id}"
            lore_preview = lore[:100] + ("..." if len(lore) > 100 else "")
            embed.add_field(name=name, value=lore_preview, inline=False)
        
        await interaction.followup.send(embed=embed)

# MEMORY COMMANDS

@tree.command(name="memory_generate", description="Generate a memory summary from recent messages (context-aware)")
async def generate_memory(interaction: discord.Interaction, num_messages: int):
    """Generate and save memory summary from recent conversation"""
    await interaction.response.defer(ephemeral=True)
    
    if not (1 <= num_messages <= 100):
        await interaction.followup.send("Number of messages must be between 1 and 100.")
        return
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    try:
        # Check if there's conversation history first
        history = get_conversation_history(interaction.channel.id)
        if not history:
            await interaction.followup.send("❌ **No conversation history found!**\n"
                                           "Start a conversation with the bot first, then try generating a memory.")
            return
        
        if len(history) < num_messages:
            await interaction.followup.send(f"⚠️ **Only {len(history)} messages available in history.**\n"
                                           f"Generating memory from all available messages instead of {num_messages}.")
        
        async with interaction.channel.typing():
            if is_dm:
                user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
                memory_summary = await generate_memory_summary(
                    interaction.channel.id, 
                    num_messages, 
                    guild=None, 
                    user_id=interaction.user.id,
                    username=user_name
                )
                context = "DM conversation"
            else:
                memory_summary = await generate_memory_summary(
                    interaction.channel.id, 
                    num_messages, 
                    interaction.guild
                )
                context = "server conversation"
        
        # Check if memory generation failed
        if not memory_summary:
            await interaction.followup.send("❌ **Failed to generate memory summary: No response from AI**")
            return
        elif memory_summary.startswith("❌") or memory_summary.startswith("Error"):
            await interaction.followup.send(f"❌ **Failed to generate memory summary:**\n{memory_summary}")
            return
        
        # Save the memory
        if is_dm:
            memory_index = memory_manager.save_dm_memory(interaction.user.id, memory_summary)
        else:
            memory_index = memory_manager.save_memory(interaction.guild.id, memory_summary)
        
        embed = discord.Embed(
            title="🧠 Generated and Saved Memory",
            description=f"Summary of the last {min(num_messages, len(history))} messages from your {context} (Memory #{memory_index + 1}):",
            color=0x9932cc
        )
        
        # Handle long summaries
        if len(memory_summary) > 1024:
            parts = [memory_summary[i:i+1024] for i in range(0, len(memory_summary), 1024)]
            for i, part in enumerate(parts):
                field_name = f"Memory Summary (Part {i+1})" if len(parts) > 1 else "Memory Summary"
                embed.add_field(name=field_name, value=part, inline=False)
        else:
            embed.add_field(name="Memory Summary", value=memory_summary, inline=False)
        
        embed.set_footer(text="✅ Memory saved! The bot will recall this when relevant topics come up.")
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        # print(f"Error in memory_generate command: {e}")
        # print(f"Full traceback: {error_details}")
        await interaction.followup.send(f"❌ **Error generating memory:** {str(e)}\n"
                                       f"Check the console for detailed error information.")

@tree.command(name="memory_save", description="Manually save a memory summary (context-aware)")
async def save_memory(interaction: discord.Interaction, summary: str):
    """Manually save a memory summary"""
    await interaction.response.defer(ephemeral=True)
    
    if len(summary) < 10:
        await interaction.followup.send("Memory summary must be at least 10 characters long.")
        return
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        memory_index = memory_manager.save_dm_memory(interaction.user.id, summary)
        context = "DM"
    else:
        memory_index = memory_manager.save_memory(interaction.guild.id, summary)
        context = "server"
    
    summary_preview = summary[:200] + ('...' if len(summary) > 200 else '')
    await interaction.followup.send(f"✅ **{context.title()} memory saved!** (Memory #{memory_index + 1})\n\n**Summary:** {summary_preview}")

@tree.command(name="memory_list", description="View all saved memories (context-aware)")
async def list_memories(interaction: discord.Interaction):
    """Display all saved memories"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        memories = memory_manager.get_all_dm_memories(interaction.user.id)
        context = "DM"
        location = "your DMs"
    else:
        memories = memory_manager.get_all_memories(interaction.guild.id)
        context = "server"
        location = interaction.guild.name
    
    if not memories:
        await interaction.followup.send(f"❌ **No {context} memories saved yet!**\n"
                                       f"Use `/memory_generate` or `/memory_save` to create some.")
        return
    
    embed = discord.Embed(
        title=f"🧠 {context.title()} Memory Bank",
        description=f"Saved memories for {location}:",
        color=0x9932cc
    )
    
    # Show last 10 memories with numbers
    start_index = max(0, len(memories) - 10)
    for i, memory in enumerate(memories[start_index:], start=start_index + 1):
        memory_text = memory["memory"][:100] + ("..." if len(memory["memory"]) > 100 else "")
        keywords = ", ".join(memory["keywords"][:5])
        embed.add_field(
            name=f"Memory #{i}",
            value=f"**Summary:** {memory_text}\n**Keywords:** {keywords}",
            inline=False
        )
    
    if len(memories) > 10:
        embed.set_footer(text=f"Showing last 10 memories out of {len(memories)} total\nUse /memory_view <number> to see full details")
    else:
        embed.set_footer(text="Use /memory_view <number> to see full details | /memory_edit <number> to edit | /memory_delete <number> to delete")
    
    await interaction.followup.send(embed=embed)

@tree.command(name="memory_clear", description="Delete all saved memories (context-aware)")
async def clear_memories(interaction: discord.Interaction):
    """Delete all memories"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        memory_count = len(memory_manager.get_all_dm_memories(interaction.user.id))
        context = "DM"
    else:
        memory_count = len(memory_manager.get_all_memories(interaction.guild.id))
        context = "server"
    
    if memory_count == 0:
        await interaction.followup.send(f"❌ **No {context} memories to clear.**")
        return
    
    if is_dm:
        memory_manager.delete_all_dm_memories(interaction.user.id)
    else:
        memory_manager.delete_all_memories(interaction.guild.id)
    
    await interaction.followup.send(f"🗑️ **Cleared all {memory_count} {context} memories!**\n"
                                   f"The bot's {context} memory has been wiped clean.")

@tree.command(name="memory_edit", description="Edit a specific saved memory (context-aware)")
async def edit_memory(interaction: discord.Interaction, memory_number: int, new_summary: str):
    """Edit a specific memory by number"""
    await interaction.response.defer(ephemeral=True)
    
    if len(new_summary) < 10:
        await interaction.followup.send("Memory summary must be at least 10 characters long.")
        return
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    # Convert to 0-based index
    memory_index = memory_number - 1
    
    if is_dm:
        # DM mode
        old_memory = memory_manager.get_dm_memory_by_index(interaction.user.id, memory_index)
        if not old_memory:
            total_memories = len(memory_manager.get_all_dm_memories(interaction.user.id))
            await interaction.followup.send(f"❌ **DM Memory #{memory_number} not found!**\n"
                                           f"Available memories: 1-{total_memories}\n"
                                           f"Use `/memory_list` to see all memories.")
            return
        
        success = memory_manager.edit_dm_memory(interaction.user.id, memory_index, new_summary)
        context = "DM"
    else:
        # Server mode
        old_memory = memory_manager.get_memory_by_index(interaction.guild.id, memory_index)
        if not old_memory:
            total_memories = len(memory_manager.get_all_memories(interaction.guild.id))
            await interaction.followup.send(f"❌ **Server Memory #{memory_number} not found!**\n"
                                           f"Available memories: 1-{total_memories}\n"
                                           f"Use `/memory_list` to see all memories.")
            return
        
        success = memory_manager.edit_memory(interaction.guild.id, memory_index, new_summary)
        context = "Server"
    
    if success:
        old_preview = old_memory["memory"][:100] + ('...' if len(old_memory["memory"]) > 100 else '')
        new_preview = new_summary[:100] + ('...' if len(new_summary) > 100 else '')
        
        embed = discord.Embed(
            title=f"🧠 {context} Memory Updated!",
            description=f"Updated Memory #{memory_number}",
            color=0x9932cc
        )
        embed.add_field(name="Previous Memory", value=old_preview, inline=False)
        embed.add_field(name="New Memory", value=new_preview, inline=False)
        
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ Failed to edit memory.")

@tree.command(name="memory_delete", description="Delete a specific saved memory (context-aware)")
async def delete_memory(interaction: discord.Interaction, memory_number: int):
    """Delete a specific memory by number"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    # Convert to 0-based index
    memory_index = memory_number - 1
    
    if is_dm:
        # DM mode
        memory_to_delete = memory_manager.get_dm_memory_by_index(interaction.user.id, memory_index)
        if not memory_to_delete:
            total_memories = len(memory_manager.get_all_dm_memories(interaction.user.id))
            await interaction.followup.send(f"❌ **DM Memory #{memory_number} not found!**\n"
                                           f"Available memories: 1-{total_memories}\n"
                                           f"Use `/memory_list` to see all memories.")
            return
        
        success = memory_manager.delete_dm_memory(interaction.user.id, memory_index)
        context = "DM"
    else:
        # Server mode
        memory_to_delete = memory_manager.get_memory_by_index(interaction.guild.id, memory_index)
        if not memory_to_delete:
            total_memories = len(memory_manager.get_all_memories(interaction.guild.id))
            await interaction.followup.send(f"❌ **Server Memory #{memory_number} not found!**\n"
                                           f"Available memories: 1-{total_memories}\n"
                                           f"Use `/memory_list` to see all memories.")
            return
        
        success = memory_manager.delete_memory(interaction.guild.id, memory_index)
        context = "Server"
    
    if success:
        deleted_preview = memory_to_delete["memory"][:150] + ('...' if len(memory_to_delete["memory"]) > 150 else '')
        await interaction.followup.send(f"🗑️ **{context} Memory #{memory_number} deleted!**\n\n"
                                       f"**Deleted memory:** {deleted_preview}")
    else:
        await interaction.followup.send("❌ Failed to delete memory.")

@tree.command(name="memory_view", description="View a specific memory in full detail (context-aware)")
async def view_memory(interaction: discord.Interaction, memory_number: int):
    """View a specific memory in detail"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    # Convert to 0-based index
    memory_index = memory_number - 1
    
    if is_dm:
        # DM mode
        memory = memory_manager.get_dm_memory_by_index(interaction.user.id, memory_index)
        if not memory:
            total_memories = len(memory_manager.get_all_dm_memories(interaction.user.id))
            await interaction.followup.send(f"❌ **DM Memory #{memory_number} not found!**\n"
                                           f"Available memories: 1-{total_memories}\n"
                                           f"Use `/memory_list` to see all memories.")
            return
        context = "DM"
    else:
        # Server mode
        memory = memory_manager.get_memory_by_index(interaction.guild.id, memory_index)
        if not memory:
            total_memories = len(memory_manager.get_all_memories(interaction.guild.id))
            await interaction.followup.send(f"❌ **Server Memory #{memory_number} not found!**\n"
                                           f"Available memories: 1-{total_memories}\n"
                                           f"Use `/memory_list` to see all memories.")
            return
        context = "Server"
    
    embed = discord.Embed(
        title=f"🧠 {context} Memory #{memory_number}",
        description=memory["memory"],
        color=0x9932cc
    )
    
    keywords = ", ".join(memory["keywords"][:10])  # Show first 10 keywords
    if len(memory["keywords"]) > 10:
        keywords += "..."
    
    embed.add_field(name="Keywords", value=keywords, inline=False)
    
    # Format timestamp
    timestamp = datetime.datetime.fromtimestamp(memory["timestamp"])
    embed.set_footer(text=f"Created: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    await interaction.followup.send(embed=embed)

# UTILITY COMMANDS

@tree.command(name="delete_messages", description="Delete the bot's last N messages from this channel/DM")
async def delete_messages(interaction: discord.Interaction, number: int):
    """Delete bot's last N logical messages from channel or DM"""
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    # Set limits based on channel type
    if is_dm:
        max_number = 20
        if not (1 <= number <= max_number):
            await interaction.response.send_message(f"Number must be between 1 and {max_number} for DMs.", ephemeral=True)
            return
    else:
        max_number = 50
        if not (1 <= number <= max_number):
            await interaction.response.send_message(f"Number must be between 1 and {max_number} for servers.", ephemeral=True)
            return
        
        # Check permissions for servers
        if not interaction.channel.permissions_for(interaction.guild.me).manage_messages:
            await interaction.response.send_message("❌ I don't have permission to delete messages in this channel.", ephemeral=True)
            return
    
    if is_dm:
        # For DMs: Defer ephemerally and delete directly
        await interaction.response.defer(ephemeral=True)
        
        # Track deleted messages for DM history loading
        if interaction.user.id not in recently_deleted_dm_messages:
            recently_deleted_dm_messages[interaction.user.id] = set()
        
        # Collect message IDs before deletion for tracking
        bot_messages_to_delete = []
        async for message in interaction.channel.history(limit=200):
            if message.author == client.user and len(message.content.strip()) > 0:
                bot_messages_to_delete.append(message)
                if len(bot_messages_to_delete) >= number * 3:  # Get extra in case of multipart
                    break
        
        # Perform deletion directly
        deleted_count = await delete_bot_messages(interaction.channel, number, set())
        
        # Track the deleted message IDs
        for msg in bot_messages_to_delete[:deleted_count * 2]:  # Approximate tracking
            recently_deleted_dm_messages[interaction.user.id].add(msg.id)
        
        # Follow up with result (ephemeral)
        if deleted_count > 0:
            await interaction.followup.send(f"✅ Deleted {deleted_count} logical message(s)!")
        else:
            await interaction.followup.send("❌ No messages found to delete.")
    else:
        # For servers: Send public status message first
        await interaction.response.send_message(f"🗑️ Deleting {number} of my logical messages...")
        
        # Get the response message to exclude it
        response_msg = await interaction.original_response()
        exclude_ids = {response_msg.id}
        
        # Perform deletion
        deleted_count = await delete_bot_messages(interaction.channel, number, exclude_ids)
        
        # Edit the status message with result
        if deleted_count > 0:
            await interaction.edit_original_response(content=f"✅ Deleted {deleted_count} logical message(s)!")
        else:
            await interaction.edit_original_response(content="❌ No messages found to delete (or permission denied).")
        
        # Delete the status message after a few seconds
        await asyncio.sleep(3)
        try:
            await response_msg.delete()
        except:
            pass

@tree.command(name="clear", description="Clear the conversation history for this specific channel/DM")
async def clear(interaction: discord.Interaction):
    """Clear conversation history for current channel"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    channel_id = interaction.channel.id
    
    # Check if DM full history is enabled (which would reload history anyway)
    dm_full_history_warning = ""
    if is_dm and dm_manager.is_dm_full_history_enabled(interaction.user.id):
        dm_full_history_warning = "\n\n⚠️ **Note:** You have DM full history loading enabled, so the bot will reload our conversation history on the next message. Use `/dm_history_toggle false` first if you want a true fresh start."
    
    # Clear stored conversation history
    history_cleared = False
    if channel_id in conversations:
        del conversations[channel_id]
        history_cleared = True
    
    # Clear recent participants (server only)
    participants_cleared = False
    if interaction.guild and channel_id in recent_participants:
        del recent_participants[channel_id]
        participants_cleared = True
    
    # Clear multipart response tracking
    multipart_cleared = False
    if channel_id in multipart_responses:
        del multipart_responses[channel_id]
        multipart_cleared = True
    
    if channel_id in multipart_response_counter:
        del multipart_response_counter[channel_id]
    
    # Provide detailed feedback
    if history_cleared or participants_cleared or multipart_cleared:
        cleared_items = []
        if history_cleared:
            cleared_items.append("conversation history")
        if participants_cleared:
            cleared_items.append("participant tracking")
        if multipart_cleared:
            cleared_items.append("message tracking")
        
        context = "DM" if is_dm else "channel"
        await interaction.followup.send(f"✅ **{context.title()} memory cleared!**\n\n"
                                       f"**Cleared:** {', '.join(cleared_items)}\n"
                                       f"The bot will start fresh from the next message.{dm_full_history_warning}")
    else:
        context = "DM" if is_dm else "channel"
        await interaction.followup.send(f"✅ **No {context} memory to clear!**\n"
                                       f"This {context} was already starting fresh.{dm_full_history_warning}")

# FUN COMMANDS

@tree.command(name="kiss", description="Give the bot a kiss! 💋")
async def kiss_command(interaction: discord.Interaction):
    """User kisses the bot - bot reacts accordingly"""
    await interaction.response.defer(ephemeral=False)
    
    user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    kiss_instruction = f"[SPECIAL INSTRUCTION]: {user_name} just gave you a kiss! React naturally to being kissed by them. Be flustered, surprised, happy, aroused, or however your character would react to suddenly being kissed. NO ASTERISKS ROLEPLAY OR REACTIONS!"
    guild_id = interaction.guild.id if interaction.guild else None

    await add_to_history(interaction.channel.id, "user", f"[{user_name} used /kiss]", interaction.user.id, guild_id, user_name=user_name)
    
    async with interaction.channel.typing():
        response = await generate_response(
            interaction.channel.id, 
            f"[{user_name} used /kiss]\n{kiss_instruction}", 
            interaction.guild, 
            None,
            user_name, 
            is_dm, 
            interaction.user.id,
            None
        )
        
        await send_fun_command_response(interaction, response)

@tree.command(name="hug", description="Give the bot a warm hug! 🤗")
async def hug_command(interaction: discord.Interaction):
    """User hugs the bot - bot reacts accordingly"""
    await interaction.response.defer(ephemeral=False)
    
    user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    hug_instruction = f"[SPECIAL INSTRUCTION]: {user_name} just gave you a warm hug! React naturally to being hugged by them. Be comforted, happy, grateful, upset, or however your character would respond to a nice hug. NO ASTERISKS ROLEPLAY OR REACTIONS!"
    guild_id = interaction.guild.id if interaction.guild else None

    await add_to_history(interaction.channel.id, "user", f"[{user_name} used /hug]", interaction.user.id, guild_id, user_name=user_name)

    async with interaction.channel.typing():
        response = await generate_response(
            interaction.channel.id, 
            f"[{user_name} used /hug]\n{hug_instruction}", 
            interaction.guild, 
            None,
            user_name, 
            is_dm, 
            interaction.user.id,
            None
        )
        
        await send_fun_command_response(interaction, response)

@tree.command(name="joke", description="Ask the bot to tell you a joke! 😄")
async def joke_command(interaction: discord.Interaction):
    """Bot tells a joke"""
    await interaction.response.defer(ephemeral=False)
    
    user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    joke_instruction = f"[SPECIAL INSTRUCTION]: {user_name} wants you to tell them a joke! Tell a funny, uncensored joke that fits your character's personality. Make it entertaining and engaging. You can be witty, punny, clever, or whatever style fits your character! NO ASTERISKS ROLEPLAY OR REACTIONS!"
    guild_id = interaction.guild.id if interaction.guild else None

    await add_to_history(interaction.channel.id, "user", f"[{user_name} used /joke]", interaction.user.id, guild_id, user_name=user_name)

    async with interaction.channel.typing():
        response = await generate_response(
            interaction.channel.id, 
            f"[{user_name} used /joke]\n{joke_instruction}", 
            interaction.guild, 
            None,
            user_name, 
            is_dm, 
            interaction.user.id,
            None
        )
        
        await send_fun_command_response(interaction, response)

@tree.command(name="bonk", description="Bonk the bot's head! 🔨")
async def bonk_command(interaction: discord.Interaction):
    """Bot gets bonked"""
    await interaction.response.defer(ephemeral=False)
    
    user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    bonk_instruction = f"[SPECIAL INSTRUCTION]: {user_name} just bonked your head! React naturally to being bonked by them. Be in pain, upset, grateful, furious, or however your character would respond to a silly bonk. NO ASTERISKS ROLEPLAY OR REACTIONS!"
    guild_id = interaction.guild.id if interaction.guild else None

    await add_to_history(interaction.channel.id, "user", f"[{user_name} used /bonk]", interaction.user.id, guild_id, user_name=user_name)
    
    async with interaction.channel.typing():
        response = await generate_response(
            interaction.channel.id, 
            f"[{user_name} used /bonk]\n{bonk_instruction}", 
            interaction.guild, 
            None,
            user_name, 
            is_dm, 
            interaction.user.id,
            None
        )
        
        await send_fun_command_response(interaction, response)

@tree.command(name="bite", description="Bite the bot! Chomp! 🧛")
async def bite_command(interaction: discord.Interaction):
    """User bites the bot - bot reacts accordingly"""
    await interaction.response.defer(ephemeral=False)
    
    user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    bite_instruction = f"[SPECIAL INSTRUCTION]: {user_name} just bit you! React naturally to being bit by them. Be in pain, amused, laughing, upset, or however your character would respond to a playful bite. REMEMBER: NO ASTERISKS ROLEPLAY OR REACTIONS!"
    guild_id = interaction.guild.id if interaction.guild else None

    await add_to_history(interaction.channel.id, "user", f"[{user_name} used /bite]", interaction.user.id, guild_id, user_name=user_name)

    async with interaction.channel.typing():
        response = await generate_response(
            interaction.channel.id, 
            f"[{user_name} used /bite]\n{bite_instruction}", 
            interaction.guild, 
            None,
            user_name, 
            is_dm, 
            interaction.user.id,
            None
        )
        
        await send_fun_command_response(interaction, response)

@tree.command(name="affection", description="Ask how much the bot likes you! 💕")
async def affection_command(interaction: discord.Interaction):
    """Bot evaluates affection level based on chat history"""
    await interaction.response.defer(ephemeral=False)
    
    user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    # Get conversation history for analysis
    if is_dm and dm_manager.is_dm_full_history_enabled(interaction.user.id):
        try:
            shared_guild = get_shared_guild(interaction.user.id)
            history = await load_all_dm_history(interaction.channel, interaction.user.id, shared_guild)
        except:
            history = get_conversation_history(interaction.channel.id)
    else:
        history = get_conversation_history(interaction.channel.id)
    
    # Analyze user interactions from history
    user_messages = []
    for msg in history:
        content = msg.get("content")
        if isinstance(content, str) and msg["role"] == "user":
            if is_dm or f"{user_name}:" in content:
                user_messages.append(content)
    
    # Create interaction context
    if user_messages:
        recent_interactions = user_messages[-10:]
        interaction_summary = " | ".join([msg[:50] + "..." if len(msg) > 50 else msg for msg in recent_interactions])
        interaction_context = f"Recent interactions with {user_name}: {interaction_summary}"
    else:
        interaction_context = f"This is one of the first interactions with {user_name}."
    
    affection_instruction = f"[SPECIAL INSTRUCTION]: {user_name} wants to know how much you like them! Based on your chat history and interactions, give them a percentage score (0-100%) of how much you like them, and explain why. Be honest. Consider things like: how often you've talked, how nice they've been, shared interests, funny moments, etc. REMEMBER: NO ASTERISKS ROLEPLAY OR REACTIONS!\n{interaction_context}"
    guild_id = interaction.guild.id if interaction.guild else None

    await add_to_history(interaction.channel.id, "user", f"[{user_name} used /affection]", interaction.user.id, guild_id, user_name=user_name)
    
    async with interaction.channel.typing():
        response = await generate_response(
            interaction.channel.id, 
            f"[{user_name} used /affection]\n{affection_instruction}", 
            interaction.guild, 
            None,
            user_name, 
            is_dm, 
            interaction.user.id,
            None
        )
        
        await send_fun_command_response(interaction, response)

# DM-SPECIFIC COMMANDS

@tree.command(name="dm_toggle", description="Toggle auto check-up messages in DMs (bot will message you once if inactive for 6+ hours)")
async def dm_toggle_command(interaction: discord.Interaction, enabled: bool = None):
    """Toggle DM auto check-up feature"""
    await interaction.response.defer(ephemeral=True)
    
    if enabled is None:
        # Show current status
        current_status = dm_manager.is_dm_toggle_enabled(interaction.user.id)
        reminder_sent = dm_manager.check_up_sent.get(interaction.user.id, False)
        
        status_text = "✅ Enabled" if current_status else "❌ Disabled"
        if current_status and reminder_sent:
            status_text += " (reminder already sent this session)"
        
        await interaction.followup.send(f"**DM Auto Check-up Status:** {status_text}\n\n"
                                       f"When enabled, I'll send you **one** message if you haven't talked to me for 6+ hours.\n"
                                       f"The reminder resets when you become active again.\n\n"
                                       f"Use `/dm_toggle true` to enable or `/dm_toggle false` to disable.")
        return
    
    dm_manager.set_dm_toggle(interaction.user.id, enabled)
    
    if enabled:
        await interaction.followup.send("✅ **DM Auto Check-up Enabled!**\n\n"
                                       "I'll now send you a caring message if you haven't talked to me for 6+ hours.\n"
                                       "**Note:** Only one reminder per session - it resets when you become active again.\n"
                                       "The reminder includes context from your recent messages for continuity.\n\n"
                                       "💡 You can disable this anytime with `/dm_toggle false`")
    else:
        await interaction.followup.send("❌ **DM Auto Check-up Disabled.**\n\n"
                                       "I won't send you automatic check-up messages anymore.\n"
                                       "You can re-enable this anytime with `/dm_toggle true`")

@tree.command(name="dm_personality_list", description="List all personalities available for your DMs")
async def dm_personality_list(interaction: discord.Interaction):
    """List all personalities available from shared servers"""
    await interaction.response.defer(ephemeral=True)
    
    user_id = interaction.user.id
    available_personalities = {}
    
    shared_guilds_count = 0
    # Collect personalities from all shared guilds
    for guild in client.guilds:
        # Try both methods to find the member
        member = guild.get_member(user_id)
        if not member:
            try:
                member = await guild.fetch_member(user_id)
            except (discord.NotFound, discord.Forbidden):
                member = None
        
        if member:  # User is in this guild
            shared_guilds_count += 1
            
            # Get current server personality
            current_personality = guild_personalities.get(guild.id, "default")
            
            # Add the active personality for this guild
            if current_personality == "default":
                display_name = DEFAULT_PERSONALITIES["default"]["name"]
                prompt = DEFAULT_PERSONALITIES["default"]["prompt"]
            elif guild.id in custom_personalities and current_personality in custom_personalities[guild.id]:
                personality_data = custom_personalities[guild.id][current_personality]
                display_name = personality_data["name"]
                prompt = personality_data["prompt"]
            else:
                display_name = "Unknown"
                prompt = "Personality data not found"
            
            available_personalities[guild.id] = {
                "name": display_name,
                "personality_key": current_personality,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "guild_name": guild.name,
            }
    
    if shared_guilds_count == 0:
        await interaction.followup.send("❌ **No shared servers found!**\n"
                                       "Make sure you're in a server with the bot and that the bot has proper permissions.\n\n"
                                       f"**Debug info:** Bot is in {len(client.guilds)} servers total.")
        return
    
    embed = discord.Embed(
        title="🎭 Available DM Personalities",
        description=f"Your DMs will automatically use personalities from servers you share with the bot.\nFound {shared_guilds_count} shared server(s):",
        color=0x9932cc
    )
    
    # Add personalities from each shared server
    for guild_id, data in available_personalities.items():
        field_name = f"{data['guild_name']}"
        field_value = f"**{data['name']}** (`{data['personality_key']}`)\n{data['prompt']}"
        embed.add_field(name=field_name, value=field_value, inline=False)
    
    embed.set_footer(text="💡 DM personality is automatically determined by your shared servers!\nUse /personality_set in servers to change personalities.")
    await interaction.followup.send(embed=embed)

@tree.command(name="dm_personality_set", description="Choose which server's personality to use in your DMs")
async def dm_personality_set(interaction: discord.Interaction, server_name: str = None):
    """Set which server's personality to use in DMs"""
    await interaction.response.defer(ephemeral=True)
    
    user_id = interaction.user.id
    
    # Collect all shared guilds and their personalities
    shared_guilds = {}
    for guild in client.guilds:
        # Try both methods to find the member
        member = guild.get_member(user_id)
        if not member:
            try:
                member = await guild.fetch_member(user_id)
            except (discord.NotFound, discord.Forbidden):
                member = None
        
        if member:  # User is in this guild
            current_personality = guild_personalities.get(guild.id, "default")
            
            # Get personality details
            if current_personality == "default":
                display_name = DEFAULT_PERSONALITIES["default"]["name"]
            elif guild.id in custom_personalities and current_personality in custom_personalities[guild.id]:
                personality_data = custom_personalities[guild.id][current_personality]
                display_name = personality_data["name"]
            else:
                display_name = "Unknown"
            
            shared_guilds[guild.name.lower()] = {
                "guild_id": guild.id,
                "guild_name": guild.name,
                "personality_key": current_personality,
                "personality_name": display_name
            }
    
    if not shared_guilds:
        await interaction.followup.send("❌ **No shared servers found!**\n"
                                       "Make sure you're in a server with the bot.")
        return
    
    # If no server name provided, show available options
    if server_name is None:
        embed = discord.Embed(
            title="🎭 Choose DM Personality",
            description="Select which server's personality to use in your DMs:",
            color=0x9932cc
        )
        
        # Get current setting
        current_guild_id = dm_manager.dm_personalities.get(user_id, (None, None))[0]
        current_server = None
        if current_guild_id:
            for guild_data in shared_guilds.values():
                if guild_data["guild_id"] == current_guild_id:
                    current_server = guild_data["guild_name"]
                    break
        
        if current_server:
            embed.add_field(
                name="Current Setting",
                value=f"Using personality from **{current_server}**",
                inline=False
            )
        else:
            embed.add_field(
                name="Current Setting",
                value="Using automatic selection (first shared server found)",
                inline=False
            )
        
        # List available servers
        server_list = []
        for guild_data in shared_guilds.values():
            server_list.append(f"• **{guild_data['guild_name']}** - {guild_data['personality_name']} (`{guild_data['personality_key']}`)")
        
        embed.add_field(
            name="Available Servers",
            value="\n".join(server_list),
            inline=False
        )
        
        embed.set_footer(text="Use /dm_personality_set <server_name> to choose\nUse /dm_personality_reset to go back to automatic")
        await interaction.followup.send(embed=embed)
        return
    
    # Find the server by name (case-insensitive)
    server_name_lower = server_name.lower()
    selected_guild = None
    
    # Try exact match first
    if server_name_lower in shared_guilds:
        selected_guild = shared_guilds[server_name_lower]
    else:
        # Try partial match
        for guild_name, guild_data in shared_guilds.items():
            if server_name_lower in guild_name:
                selected_guild = guild_data
                break
    
    if not selected_guild:
        available_servers = [guild_data["guild_name"] for guild_data in shared_guilds.values()]
        await interaction.followup.send(f"❌ **Server not found!**\n\n"
                                       f"Available servers: {', '.join(available_servers)}\n"
                                       f"Use `/dm_personality_set` without arguments to see all options.")
        return
    
    # Set the DM personality
    dm_manager.dm_personalities[user_id] = (selected_guild["guild_id"], selected_guild["personality_key"])
    dm_manager.save_data()
    
    await interaction.followup.send(f"✅ **DM Personality Set!**\n\n"
                                   f"**Server:** {selected_guild['guild_name']}\n"
                                   f"**Personality:** {selected_guild['personality_name']} (`{selected_guild['personality_key']}`)\n\n"
                                   f"💬 Your DMs will now use this personality!\n"
                                   f"💡 Use `/dm_personality_reset` to go back to automatic selection.")

@tree.command(name="dm_personality_reset", description="Reset DM personality to automatic selection")
async def dm_personality_reset(interaction: discord.Interaction):
    """Reset DM personality to automatic selection"""
    await interaction.response.defer(ephemeral=True)
    
    user_id = interaction.user.id
    
    if user_id in dm_manager.dm_personalities:
        del dm_manager.dm_personalities[user_id]
        dm_manager.save_data()
        await interaction.followup.send("✅ **DM Personality Reset!**\n\n"
                                       "Your DMs will now automatically use the personality from the first shared server found.\n"
                                       "Use `/dm_personality_set` to choose a specific server's personality.")
    else:
        await interaction.followup.send("✅ **Already using automatic selection!**\n\n"
                                       "Your DMs automatically use personalities from shared servers.\n"
                                       "Use `/dm_personality_set` to choose a specific server's personality.")

@dm_personality_set.autocomplete('server_name')
async def server_name_autocomplete(interaction: discord.Interaction, current: str):
    """Autocomplete for server names"""
    user_id = interaction.user.id
    shared_servers = []
    
    for guild in client.guilds:
        member = guild.get_member(user_id)
        if not member:
            try:
                member = await guild.fetch_member(user_id)
            except (discord.NotFound, discord.Forbidden):
                continue
        
        if member and current.lower() in guild.name.lower():
            shared_servers.append(app_commands.Choice(name=guild.name, value=guild.name))
    
    return shared_servers[:25]  # Discord limits to 25 choices

@tree.command(name="dm_history_toggle", description="Toggle loading full DM conversation history (DMs only)")
async def dm_history_toggle(interaction: discord.Interaction, enabled: bool = None):
    """Toggle full DM history loading"""
    await interaction.response.defer(ephemeral=True)
    
    if not isinstance(interaction.channel, discord.DMChannel):
        await interaction.followup.send("❌ This command only works in DMs!")
        return
    
    if enabled is None:
        # Show current status
        current_status = dm_manager.is_dm_full_history_enabled(interaction.user.id)
        status_text = "✅ Enabled" if current_status else "❌ Disabled"
        await interaction.followup.send(f"**DM Full History Loading:** {status_text}\n\n"
                                       f"When enabled, I'll load our entire conversation history (up to token limits) so I remember everything we've talked about, even across bot restarts.\n\n"
                                       f"**Privacy Note:** This only reads existing DM messages - nothing is saved to files.\n\n"
                                       f"Use `/dm_history_toggle true` to enable or `/dm_history_toggle false` to disable.")
        return
    
    dm_manager.set_dm_full_history(interaction.user.id, enabled)
    
    if enabled:
        await interaction.followup.send("✅ **DM Full History Loading Enabled!**\n\n"
                                       "I'll now load our complete conversation history each time we chat, so I remember everything we've discussed.\n\n"
                                       "**Benefits:**\n"
                                       "• I remember past conversations even after restarts\n"
                                       "• Better context and continuity\n"
                                       "• More personalized responses\n\n"
                                       "**Privacy:** Only reads existing messages, nothing is saved to files.\n\n"
                                       "💡 You can disable this anytime with `/dm_history_toggle false`")
    else:
        await interaction.followup.send("❌ **DM Full History Loading Disabled.**\n\n"
                                       "I'll only remember our recent conversation (standard behavior).\n"
                                       "You can re-enable this anytime with `/dm_history_toggle true`")

@tree.command(name="dm_edit_last", description="Edit the bot's last message in this DM (DMs only)")
async def dm_edit_last_message(interaction: discord.Interaction, new_content: str):
    """Edit the bot's last logical message (DMs only)"""
    await interaction.response.defer(ephemeral=True)
    
    # Check if this is a DM
    if not isinstance(interaction.channel, discord.DMChannel):
        await interaction.followup.send("❌ This command only works in DMs! Bot message editing is not allowed in servers to maintain natural conversation flow.")
        return
    
    try:
        # Get the bot's last logical message (handles multipart responses)
        messages_to_edit, original_content = await get_bot_last_logical_message(interaction.channel)
        
        if not messages_to_edit:
            await interaction.followup.send("❌ No recent bot message found to edit!")
            return
        
        # Delete ALL the old messages (entire logical response)
        for msg in messages_to_edit:
            try:
                await msg.delete()
            except:
                pass
        
        # Send the new content (split by newlines if needed)
        message_parts = split_message_by_newlines(new_content)
        
        sent_messages = []
        if len(message_parts) > 1:
            for part in message_parts:
                if len(part) > 4000:
                    for i in range(0, len(part), 4000):
                        sent_msg = await interaction.channel.send(part[i:i+4000])
                        sent_messages.append(sent_msg)
                else:
                    sent_msg = await interaction.channel.send(part)
                    sent_messages.append(sent_msg)
        else:
            if len(new_content) > 4000:
                for i in range(0, len(new_content), 4000):
                    sent_msg = await interaction.channel.send(new_content[i:i+4000])
                    sent_messages.append(sent_msg)
            else:
                sent_msg = await interaction.channel.send(new_content)
                sent_messages.append(sent_msg)
        
        # Store as multipart response if needed
        if len(sent_messages) > 1:
            store_multipart_response(interaction.channel.id, [msg.id for msg in sent_messages], new_content)
        
        # Update conversation history (for DMs, check if using full history)
        if dm_manager.is_dm_full_history_enabled(interaction.user.id):
            await interaction.followup.send("✅ **Message edited!** The edit is automatically reflected in conversation history.")
        else:
            # Update stored conversation history
            if interaction.channel.id in conversations:
                history = conversations[interaction.channel.id]
                
                # Find the most recent assistant message and update it
                for i in range(len(history) - 1, -1, -1):
                    if history[i]["role"] == "assistant":
                        history[i]["content"] = new_content
                        break
            
            await interaction.followup.send("✅ **Message edited and conversation history updated!**")
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error editing message: {str(e)}")

@tree.command(name="dm_regenerate", description="Regenerate the bot's last response with a different answer (DMs only)")
async def dm_regenerate_last_response(interaction: discord.Interaction):
    """Regenerate the bot's last logical message with a new response (DMs only)"""
    await interaction.response.defer(ephemeral=True)
    
    # Check if this is a DM
    if not isinstance(interaction.channel, discord.DMChannel):
        await interaction.followup.send("❌ This command only works in DMs! Message regeneration is not allowed in servers to maintain natural conversation flow.")
        return
    
    try:
        # Get the bot's last logical message
        messages_to_delete, original_content = await get_bot_last_logical_message(interaction.channel)
        
        if not messages_to_delete:
            await interaction.followup.send("❌ No recent bot message found to regenerate!")
            return
        
        # Find the user message that triggered this response
        user_message_before = None
        oldest_bot_message = min(messages_to_delete, key=lambda m: m.created_at)
        
        async for message in interaction.channel.history(limit=50, before=oldest_bot_message):
            if message.author != client.user:
                user_message_before = message
                break
        
        if not user_message_before:
            await interaction.followup.send("❌ Couldn't find the user message that triggered the bot's response!")
            return
        
        # Delete all the old bot messages
        for msg in messages_to_delete:
            try:
                await msg.delete()
            except:
                pass
        
        # Get user info
        user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
        guild_id = None
        
        # Get guild from selected server or shared guild
        selected_guild_id = dm_server_selection.get(interaction.user.id)
        if selected_guild_id:
            guild = client.get_guild(selected_guild_id)
            guild_id = selected_guild_id
        else:
            guild = get_shared_guild(interaction.user.id)
            if guild:
                guild_id = guild.id
        
        # If using regular history, remove the old bot response
        if not dm_manager.is_dm_full_history_enabled(interaction.user.id):
            if interaction.channel.id in conversations:
                history = conversations[interaction.channel.id]
                if history and history[-1]["role"] == "assistant":
                    history.pop()
        
        # Generate a new response
        async with interaction.channel.typing():
            new_response = await generate_response(
                interaction.channel.id,
                user_message_before.content,
                guild,
                user_message_before.attachments,
                user_name,
                is_dm=True,
                user_id=interaction.user.id,
                original_message=user_message_before
            )
        
        # Send the new response
        if new_response:
            message_parts = split_message_by_newlines(new_response)
            
            sent_messages = []
            if len(message_parts) > 1:
                for part in message_parts:
                    if len(part) > 4000:
                        for i in range(0, len(part), 4000):
                            sent_msg = await interaction.channel.send(part[i:i+4000])
                            sent_messages.append(sent_msg)
                    else:
                        sent_msg = await interaction.channel.send(part)
                        sent_messages.append(sent_msg)
            else:
                if len(new_response) > 4000:
                    for i in range(0, len(new_response), 4000):
                        sent_msg = await interaction.channel.send(new_response[i:i+4000])
                        sent_messages.append(sent_msg)
                else:
                    sent_msg = await interaction.channel.send(new_response)
                    sent_messages.append(sent_msg)
            
            # Store as multipart response if needed
            if len(sent_messages) > 1:
                store_multipart_response(interaction.channel.id, [msg.id for msg in sent_messages], new_response)
        
        await interaction.followup.send("✅ **Response regenerated!** I've created a new response to your previous message.")
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error regenerating response: {str(e)}")

@tree.command(name="dm_enable", description="Enable/disable DMs with the bot for all server members (Admin only)")
async def toggle_dm_enable(interaction: discord.Interaction, enabled: bool = None):
    """Enable or disable DMs for all server members"""
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        await interaction.followup.send("❌ This command can only be used in servers!")
        return
    
    # Check admin permissions
    if not check_admin_permissions(interaction):
        await interaction.followup.send("❌ Only administrators can control DM settings!")
        return
    
    if enabled is None:
        # Show current status
        current_status = guild_dm_enabled.get(interaction.guild.id, True)  # Default to enabled
        status_text = "✅ Enabled" if current_status else "❌ Disabled"
        
        await interaction.followup.send(f"**DM Status for {interaction.guild.name}:** {status_text}\n\n"
                                       f"When enabled, server members can DM the bot directly.\n"
                                       f"When disabled, the bot will inform users that DMs are not allowed and to contact server admins.\n\n"
                                       f"Use `/dm_enable true` to enable or `/dm_enable false` to disable.")
        return
    
    # Set the DM enabled status
    guild_dm_enabled[interaction.guild.id] = enabled
    save_json_data(DM_ENABLED_FILE, guild_dm_enabled)
    
    if enabled:
        await interaction.followup.send(f"✅ **DMs Enabled for {interaction.guild.name}!**\n\n"
                                       f"Server members can now DM the bot directly.\n"
                                       f"💡 Use `/dm_enable false` to disable DMs in the future.")
    else:
        await interaction.followup.send(f"❌ **DMs Disabled for {interaction.guild.name}!**\n\n"
                                       f"Server members will be told DMs are disabled and to contact server admins.\n"
                                       f"💡 Use `/dm_enable true` to re-enable DMs in the future.")

@tree.command(name="lore_auto_update", description="Let the bot update lore entries based on what it learned (Admin only)")
async def lore_auto_update(interaction: discord.Interaction, member: str = None):
    """Let the bot automatically update lore based on conversation history"""
    await interaction.response.defer(ephemeral=True)
    
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    
    if is_dm:
        # DM mode - update personal lore
        user_id = interaction.user.id  # Define user_id here
        history = get_conversation_history(interaction.channel.id)
        if not history:
            await interaction.followup.send("❌ No conversation history found to analyze!")
            return
        
        # Get existing lore
        existing_lore = lore_book.get_dm_entry(user_id)
        
        # Create instruction for AI to analyze and update lore
        lore_instruction = f"""Analyze the conversation history and update the user's lore entry with new information you've learned about them.

Current lore: {existing_lore if existing_lore else "No existing lore"}

Instructions:
- Extract key information about the user (personality, interests, background, preferences, etc.).
- Merge new information with existing lore, don't duplicate.
- Keep it concise and relevant for future conversations.
- Format as a brief character description (max 300 characters).
- Only include factual information explicitly mentioned by the user."""

        # Generate updated lore
        update_prompt = f"Based on our conversation, create an updated lore entry about {interaction.user.display_name}."
        
        # Use the AI to generate the update
        temp_messages = [{"role": "user", "content": update_prompt}]
        
        guild = get_shared_guild(user_id)
        guild_id = guild.id if guild else None

        # Use appropriate guild ID for temperature
        temp_guild_id = guild.id if guild else (dm_server_selection.get(user_id) if user_id else None)
        if not temp_guild_id and user_id:
            shared_guild = get_shared_guild(user_id)
            temp_guild_id = shared_guild.id if shared_guild else None
        
        updated_lore = await ai_manager.generate_response(
            messages=temp_messages,
            system_prompt=lore_instruction,
            temperature=get_temperature(temp_guild_id) if temp_guild_id else 1.0,
            user_id=user_id,
            guild_id=guild_id,
            is_dm=is_dm,
            max_tokens=500
        )
        
        if updated_lore and not updated_lore.startswith("❌"):
            # Show preview and confirm
            embed = discord.Embed(
                title="📖 Auto-Generated Lore Update",
                description="Based on our conversation, here's what I've learned:",
                color=0x00ff99
            )
            
            if existing_lore:
                embed.add_field(name="Current Lore", value=existing_lore[:300] + "..." if len(existing_lore) > 300 else existing_lore, inline=False)
            
            embed.add_field(name="Updated Lore", value=updated_lore[:300] + "..." if len(updated_lore) > 300 else updated_lore, inline=False)
            
            # Actually update the lore
            lore_book.add_dm_entry(user_id, updated_lore)
            
            embed.set_footer(text="✅ Lore has been updated! Use /lore_view to see the full entry.")
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send("❌ Failed to generate lore update.")
            
    else:
        # Server mode
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can use auto-update in servers!")
            return
            
        if member is None:
            await interaction.followup.send("❌ Please specify a member to update lore for.\n"
                                           "**Usage:** `/lore_auto_update <member>`")
            return
        
        # Find member (same logic as other lore commands)
        member_obj = None
        if member.startswith('<@') and member.endswith('>'):
            user_id_str = member.strip('<@!>')
            try:
                member_obj = interaction.guild.get_member(int(user_id_str))
            except ValueError:
                pass
        elif member.isdigit():
            member_obj = interaction.guild.get_member(int(member))
        else:
            member_lower = member.lower()
            for guild_member in interaction.guild.members:
                if (guild_member.name.lower() == member_lower or 
                    guild_member.display_name.lower() == member_lower):
                    member_obj = guild_member
                    break
        
        if member_obj is None:
            await interaction.followup.send(f"❌ Member '{member}' not found!")
            return
        
        # Check if member has participated in recent conversations
        if interaction.channel.id not in recent_participants or member_obj.id not in recent_participants[interaction.channel.id]:
            await interaction.followup.send(f"❌ {member_obj.display_name} hasn't participated in recent conversations in this channel!")
            return
        
        # Get conversation history
        history = get_conversation_history(interaction.channel.id)
        
        # Filter messages involving this member
        member_messages = []
        for msg in history:
            if msg["role"] == "user" and isinstance(msg["content"], str):
                if f"<@{member_obj.id}>" in msg["content"] or member_obj.display_name in msg["content"]:
                    member_messages.append(msg["content"])
        
        if not member_messages:
            await interaction.followup.send(f"❌ No messages from {member_obj.display_name} found in recent history!")
            return
        
        # Get existing lore
        existing_lore = lore_book.get_entry(interaction.guild.id, member_obj.id)
        
        # Create instruction for AI
        recent_activity = "\n".join(member_messages[-10:])  # Last 10 messages
        lore_instruction = f"""Analyze {member_obj.display_name}'s messages and update their lore entry.

Current lore: {existing_lore if existing_lore else "No existing lore"}

Recent messages from {member_obj.display_name}:
{recent_activity}

Instructions:
- Extract key information about them (personality, interests, role in server, etc.).
- Merge with existing lore, don't duplicate.
- Keep it concise and relevant.
- Format as a brief character description (max 500 characters).
- Only include factual information from their messages."""

        # Generate updated lore
        update_prompt = f"Based on the conversation, create an updated lore entry about {member_obj.display_name}."
        
        temp_messages = [{"role": "user", "content": update_prompt}]
        
        # Use the guild temperature for server mode
        temp_guild_id = interaction.guild.id
        
        updated_lore = await ai_manager.generate_response(
            messages=temp_messages,
            system_prompt=lore_instruction,
            temperature=get_temperature(temp_guild_id) if temp_guild_id else 1.0,
            guild_id=interaction.guild.id,
            is_dm=False,
            max_tokens=500
        )
        
        if updated_lore and not updated_lore.startswith("❌"):
            # Show preview
            embed = discord.Embed(
                title=f"📖 Auto-Generated Lore Update for {member_obj.display_name}",
                description="Based on recent conversations:",
                color=0x00ff99
            )
            
            if existing_lore:
                embed.add_field(name="Current Lore", value=existing_lore[:300] + "..." if len(existing_lore) > 300 else existing_lore, inline=False)
            
            embed.add_field(name="Updated Lore", value=updated_lore[:300] + "..." if len(updated_lore) > 300 else updated_lore, inline=False)
            
            # Update the lore
            lore_book.add_entry(interaction.guild.id, member_obj.id, updated_lore)
            
            embed.set_footer(text="✅ Lore has been updated!")
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send("❌ Failed to generate lore update.")

# Start the bot
if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
