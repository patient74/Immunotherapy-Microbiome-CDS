"""
Model loading and inference utilities
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

from . import config

logger = logging.getLogger(__name__)


import re
from transformers import AutoProcessor, AutoModelForImageTextToText

class MedGemmaGenerator:
    """Wrapper for MedGemma 1.5 4B model"""
    
    def __init__(self):
        logger.info(f"Loading MedGemma model: {config.MEDGEMMA_MODEL_ID}")
        
        # Use AutoProcessor (not AutoTokenizer) and AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(config.MEDGEMMA_MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.MEDGEMMA_MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if config.MEDGEMMA_DEVICE == "cuda" else None,
        )
        
        if config.MEDGEMMA_DEVICE == "cpu":
            self.model = self.model.to("cpu")
        
        self.model.eval()
        logger.info("MedGemma model loaded successfully")
    
    def _strip_thinking_block(self, text: str) -> str:
        """
        Remove the thinking/reasoning block that Gemma 3-based models emit.
        MedGemma 1.5 uses <unused94>thought...<unused95> tokens.
    
        """
        
        text = re.sub(
            r"<unused94>thought[\s\S]*?<unused95>",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"<think>[\s\S]*?</think>",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"<unused94>thought[\s\S]*$",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"<think>[\s\S]*$",
            "",
            text,
            flags=re.IGNORECASE,
        )

        return text.strip()
    
    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Generate text from prompt using MedGemma
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override default max tokens if provided
            
        Returns:
            Generated text (with thinking block removed)
        """
        gen_config = config.GENERATION_CONFIG.copy()
        if max_new_tokens:
            gen_config["max_new_tokens"] = max_new_tokens
        
        # Use proper message format for MedGemma 1.5 4B-IT
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        
        # Apply chat template properly
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config,
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else self.processor.pad_token_id,
            )
        
        # Extract only the generated portion (after the input)
        generated_tokens = outputs[0][input_len:]
        generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        
        # Strip the thinking block before returning
        generated_text = self._strip_thinking_block(generated_text)
        
        return generated_text

class EmbeddingModel:
    """Wrapper for PubMedBERT embedding model"""
    
    def __init__(self):
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_ID}")
        
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL_ID,
            device=config.EMBEDDING_DEVICE
        )
        
        logger.info("Embedding model loaded successfully")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def encode_single(self, text: str) -> List[float]:
        """
        Encode a single text to embedding
        
        Args:
            text: Text string to encode
            
        Returns:
            Embedding vector
        """
        return self.encode([text])[0]


# Global model instances (loaded once)
_medgemma_instance = None
_embedding_instance = None


def get_medgemma() -> MedGemmaGenerator:
    """Get or create MedGemma model instance"""
    global _medgemma_instance
    if _medgemma_instance is None:
        _medgemma_instance = MedGemmaGenerator()
    return _medgemma_instance


def get_embedding_model() -> EmbeddingModel:
    """Get or create embedding model instance"""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingModel()
    return _embedding_instance
