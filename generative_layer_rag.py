"""Generative Chat Layer with RAG (Retrieval-Augmented Generation)
Uses sentence-transformers for semantic search and FAISS for fast retrieval.
"""

import os
import torch
from pathlib import Path
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

# Check if FAISS is available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available. Install with: pip install faiss-cpu")

from transformers import AutoTokenizer, AutoModelForCausalLM


class GenerativeChatRAG:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_4bit: bool = False,
        kb_path: Optional[str] = None
    ):
        """
        Initialize RAG-powered generative chat with knowledge base retrieval.
        
        Args:
            model_name: HuggingFace model name for generation
            use_4bit: Whether to use 4-bit quantization (requires bitsandbytes)
            kb_path: Path to knowledge base text file
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.kb_path = kb_path
        self.knowledge_base = []
        self.embeddings = None
        self.index = None
        self.embedding_model = None
        
        # Set default KB path
        if self.kb_path is None:
            project_root = Path(__file__).parent.parent
            self.kb_path = project_root / "data" / "f1-knowledge-base.txt"
        
        # Load knowledge base and create embeddings
        if FAISS_AVAILABLE:
            self._load_knowledge_base()
            self._create_embeddings()
        else:
            print("⚠️ RAG disabled: FAISS not available")
        
        # Load generation model
        try:
            print(f"🤖 Loading generative model: {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if use_4bit and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except Exception as e:
                    print(f"⚠️ 4-bit quantization failed: {e}")
                    print("   Loading model without quantization...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
            
            self.model.eval()
            print(f"✅ Generative model loaded on {self.device}")
        
        except Exception as e:
            print(f"❌ Failed to load generative model: {e}")
            self.model = None
            self.tokenizer = None
    
    def _load_knowledge_base(self):
        """Load F1 knowledge base from text file."""
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into paragraphs (each paragraph is a knowledge chunk)
            self.knowledge_base = [p.strip() for p in content.split('\n\n') if p.strip()]
            print(f"✅ Loaded {len(self.knowledge_base)} knowledge chunks from KB")
        
        except Exception as e:
            print(f"⚠️ Could not load knowledge base: {e}")
            self.knowledge_base = []
    
    def _create_embeddings(self):
        """Create embeddings for knowledge base using sentence-transformers."""
        if not self.knowledge_base:
            print("⚠️ No knowledge base loaded, skipping embedding creation")
            return
        
        try:
            print("🔄 Creating knowledge base embeddings...")
            
            # Load sentence transformer model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create embeddings
            self.embeddings = self.embedding_model.encode(
                self.knowledge_base,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Create FAISS index for fast similarity search
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            print(f"✅ Created FAISS index with {len(self.knowledge_base)} entries")
        
        except Exception as e:
            print(f"⚠️ Failed to create embeddings: {e}")
            self.embeddings = None
            self.index = None
    
    def _retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve most relevant knowledge from KB using semantic search.
        
        Args:
            query: Question
            top_k: Number of relevant chunks to retrieve
        
        Returns:
            Combined context string
        """
        if self.index is None or self.embedding_model is None:
            return ""
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True
            ).astype('float32')
            
            # Search for similar chunks
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Retrieve and combine relevant chunks
            context_chunks = [self.knowledge_base[i] for i in indices[0]]
            context = "\n\n".join(context_chunks)
            
            return context
        
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return ""
    
    def generate_reply(
        self,
        message: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        use_rag: bool = True
    ) -> Optional[str]:
        """
        Generate a response using RAG (retrieval + generation).
        
        Args:
            message: Message
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_rag: Whether to use RAG
        
        Returns:
            Generated response or None if generation fails
        """
        if not self.is_available():
            return None
        
        try:
            # Retrieve relevant context from KB
            context = ""
            if use_rag and self.index is not None:
                context = self._retrieve_context(message, top_k=3)
                if context:
                    print(f"   📚 RAG: Retrieved {len(context)} chars of context from KB")
                else:
                    print("   ⚠️ RAG: No relevant context found in KB")
            else:
                print("   ⚠️ RAG: Disabled or index not available")
            
            # Build prompt with context
            if context:
                prompt = f"""You are an F1 expert assistant. Use the following information to answer the question accurately.

Context:
{context}

Question: {message}

Answer:"""
            else:
                prompt = f"""You are an F1 expert assistant.

Question: {message}

Answer:"""
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated answer (after "Answer:")
            if "Answer:" in full_response:
                response = full_response.split("Answer:")[-1].strip()
            else:
                response = full_response[len(prompt):].strip()
            
            # Clean up response
            response = response.split('\n')[0]
            response = response.strip()
            
            return response if response else None
        
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if generative model is loaded and ready."""
        return self.model is not None and self.tokenizer is not None


# Fallback responses for when RAG/LLM fails
FALLBACK_RESPONSES = {
    'general_question': [
        "I'm here to help with F1! Could you rephrase your question?",
        "That's an interesting F1 question! Can you be more specific?",
        "I'd love to help! Could you provide more details about what you'd like to know?",
    ]
}
