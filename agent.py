#!/usr/bin/env python

"""
Question Answering Agent Implementation

Detects and answers questions using an LLM. The agent identifies questions based on both
natural language patterns and vector similarity, then uses an LLM to generate accurate
and helpful answers.
"""

import json
import logging
import os
import re
import asyncio
import time
from typing import Dict, Any, Optional, List, Union

# For containerized agents, use the local base agent
# This avoids dependencies on the semsubscription module
try:
    # First try to import from semsubscription if available (for local development)
    from semsubscription.agents.llm_agent import LLMAgent
    from semsubscription.vector_db.database import Message
    from semsubscription.memory.vector_memory import get_memory_system
except ImportError:
    try:
        # Fall back to local agent_base for containerized environments using relative import
        from .agent_base import BaseAgent as LLMAgent  # Use BaseAgent if LLMAgent is not available
        from .message import Message  # Local implementation of Message
        # Define a dummy get_memory_system function if not available
        def get_memory_system():
            return None
    except ImportError:
        # Last resort for Docker environment with current directory
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from agent_base import BaseAgent as LLMAgent  # Use BaseAgent if LLMAgent is not available
        # Create a simple Message class if not available
        class Message:
            def __init__(self, content=""):
                self.content = content
        # Define a dummy get_memory_system function
        def get_memory_system():
            return None

logger = logging.getLogger(__name__)

class QuestionAnsweringAgent(LLMAgent):
    """
    Agent that detects and answers questions using an LLM
    
    This agent specializes in identifying questions in natural language and providing
    accurate, helpful answers using an LLM. It uses both pattern matching and
    vector similarity to determine if a message contains a question.
    """
    
    def __init__(self, agent_id=None, name=None, description=None, similarity_threshold=0.6, **kwargs):
        """
        Initialize the agent with its parameters and setup the classifier
        
        Args:
            agent_id: Optional unique identifier for the agent
            name: Optional name for the agent (defaults to class name)
            description: Optional description of the agent
            similarity_threshold: Threshold for similarity-based interest determination
        """
        # Set default name if not provided
        name = name or "Question Answering Agent"
        description = description or "Detects and answers questions using an LLM"
        
        # Call parent constructor
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            similarity_threshold=similarity_threshold,
            # Enable classifier by default for more accurate message routing
            use_classifier=True,  
            **kwargs
        )
        
        # Configure agent settings
        custom_config = self.config.get('custom', {}) if hasattr(self, 'config') else {}
        self.max_answer_length = custom_config.get('max_answer_length', 1500)
        self.answer_format = custom_config.get('answer_format', 'markdown')
        
        # Initialize question patterns
        self.question_patterns = [
            re.compile(r'^(?:what|who|where|when|why|how|is|are|can|could|would|will|should)', re.IGNORECASE),
            re.compile(r'.*\?$', re.IGNORECASE)
        ]
        
        logger.info(f"{name} agent initialized")
    
    def setup_interest_model(self):
        """
        Configure the agent's interest model with domain-specific knowledge
        """
        # Don't call super().setup_interest_model() if creating custom interest model
        
        # Get the path to the fine-tuned model
        model_path = os.path.join(os.path.dirname(__file__), "fine_tuned_model")
        logger.info(f"Using fine-tuned model from: {model_path}")
        
        try:
            # Import necessary components for fine-tuned model
            try:
                # First try importing from semsubscription
                from semsubscription.vector_db.embedding import EmbeddingEngine, InterestModel
            except ImportError:
                # Fall back to local implementation for containerized environments
                try:
                    from .interest_model import CustomInterestModel as InterestModel
                    from .embedding_engine import EmbeddingEngine
                except ImportError:
                    # Last resort fallback
                    from interest_model import CustomInterestModel as InterestModel
                    from embedding_engine import EmbeddingEngine
            
            if os.path.exists(model_path) and os.path.isdir(model_path):
                # Create embedding engine with the fine-tuned model
                embedding_engine = EmbeddingEngine(model_name=model_path)
                logger.info(f"Successfully loaded fine-tuned model with dimension: {embedding_engine.get_dimension() if hasattr(embedding_engine, 'get_dimension') else 'unknown'}")
                
                # Create interest model with the custom embedding engine
                self.interest_model = InterestModel(embedding_engine=embedding_engine)
                
                # Lower the threshold to catch more potential questions
                self.similarity_threshold = 0.5  # Lower from default
                self.interest_model.threshold = self.similarity_threshold
                logger.info(f"Set similarity threshold to {self.similarity_threshold}")
                
                # Add question-related keywords for backup matching
                self.interest_model.keywords = [
                    'question', 'answer', 'how', 'what', 'when', 'where', 'why',
                    'who', 'which', 'whose', 'explain', 'tell me', 'describe',
                    'help me understand', '?'
                ]
                
                # Get question patterns from config
                if hasattr(self, 'config') and 'question_patterns' in self.config.get('custom', {}):
                    pattern_strings = self.config['custom']['question_patterns']
                    self.question_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_strings]
                
                # Train with example questions
                examples = [
                    "What is the capital of France?",
                    "How does photosynthesis work?",
                    "Can you explain quantum computing?",
                    "Who wrote Pride and Prejudice?",
                    "When was the first computer invented?",
                    "Why is the sky blue?",
                    "Tell me about the history of chocolate",
                    "What's the difference between machine learning and AI?",
                    "How far is the moon from earth?",
                    "What are the main challenges of climate change?"
                ]
                
                # Force retraining of interest model to ensure vectors are stored properly
                logger.info(f"Training question answering agent with {len(examples)} examples using fine-tuned model")
                if hasattr(self.interest_model, 'interest_vectors'):
                    self.interest_model.interest_vectors = []  # Clear any existing vectors
                if examples and len(examples) > 0:
                    self.interest_model.train(examples, method="average")  # Use simpler averaging method
            else:
                logger.warning(f"Fine-tuned model directory not found at {model_path}, falling back to default setup")
                super().setup_interest_model()
                
        except Exception as e:
            logger.error(f"Error setting up fine-tuned model: {e}")
            # Fall back to default setup if fine-tuned model fails
            super().setup_interest_model()
            logger.warning("Fell back to default interest model setup due to error with fine-tuned model")
    
    def is_interested(self, message: Message) -> bool:
        """
        Determine if the message contains a question
        
        Args:
            message: The message to check
            
        Returns:
            True if the message contains a question, False otherwise
        """
        # Check if a classifier decision has already been made
        if hasattr(message, 'classifier_decision'):
            return message.classifier_decision
        
        # Get the message content - handle both direct strings and structured messages
        content = message.content
        if isinstance(content, dict) and 'message' in content:
            content = content['message']
        elif isinstance(content, dict) and 'content' in content:
            content = content['content']
        
        if not isinstance(content, str):
            # Try to convert to string if possible
            try:
                content = str(content)
            except:
                logger.warning(f"Cannot process non-string message content: {type(content)}")
                return False
            
        # Try pattern matching first for efficiency
        for pattern in self.question_patterns:
            if pattern.search(content):
                logger.info(f"Question pattern match: {content[:50]}...")
                return True
                
        # Try keyword matching as a fallback
        content_lower = content.lower()
        for keyword in self.interest_model.keywords:
            if keyword.lower() in content_lower:
                logger.info(f"Question keyword match on '{keyword}': {content[:50]}...")
                return True
        
        # Fall back to the standard interest determination
        return super().is_interested(message)
        

    
    async def process_message_async(self, message: Message) -> Optional[Dict[str, Any]]:
        """
        Process questions using LLM and retrieve relevant context
        
        Args:
            message: The message to process
            
        Returns:
            Response data with the answer
        """
        try:
            # Extract the question from the message
            if hasattr(message, 'content'):
                question = message.content.strip()
                message_id = getattr(message, 'id', 'unknown')
            else:
                question = message.get('content', '').strip()
                message_id = message.get('id', 'unknown')
            
            # Skip if there is no question (shouldn't happen due to interest check)
            if not question:
                return None
                
            start_time = time.time()
            logger.info(f"Processing question: {question[:100]}...")
            
            # Get relevant context from memory system if available
            memory_context = self._retrieve_relevant_context(question)
            
            # If no memory context available and LLM is available, use it directly
            if not memory_context and hasattr(self, 'llm') and self.llm:
                try:
                    result = await super().process_message_async(message)
                    result["processing_time"] = round(time.time() - start_time, 2)
                    result["source"] = "llm_direct"
                    return result
                except Exception as e:
                    logger.error(f"Error in LLM processing: {e}")
                    return self.process_message(message)  # Fall back to standard processing
                    
            # Prepare enhanced context for the LLM
            system_prompt = self._get_enhanced_system_prompt(memory_context)
            
            # Call the LLM with the enhanced prompt and context
            if hasattr(self, 'llm') and self.llm:
                try:
                    llm_response = await self.llm.complete_chat(
                        system_prompt=system_prompt,
                        messages=[{"role": "user", "content": question}],
                        temperature=self.config.get('llm', {}).get('temperature', 0.7) if hasattr(self, 'config') else 0.7,
                        max_tokens=min(self.max_answer_length, 
                                      self.config.get('llm', {}).get('max_tokens', 1000) if hasattr(self, 'config') else 1000)
                    )
                    
                    answer = llm_response.get('content', "I couldn't generate an answer at this time.")
                    
                    result = {
                        "agent": self.name,
                        "query_type": "question",
                        "question": question,
                        "answer": answer,
                        "format": self.answer_format,
                        "processing_time": round(time.time() - start_time, 2),
                        "source": "llm_with_context" if memory_context else "llm_direct"
                    }
                    
                    if memory_context:
                        result["context_sources"] = len(memory_context)
                        
                    return result
                except Exception as e:
                    logger.error(f"Error in LLM processing: {e}")
                    # Fall back to non-LLM processing
                    return self.process_message(message)
            else:
                # Fallback to non-LLM processing
                return self.process_message(message)
                
        except Exception as e:
            logger.error(f"Error in Question Answering Agent async processing: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "query": question if 'question' in locals() else "unknown query"
            }
    
    def process_message(self, message) -> Optional[Dict[str, Any]]:
        """
        Fallback for processing without LLM
        
        Args:
            message: The message to process
            
        Returns:
            Response data with a simple answer
        """
        try:
            # Handle both Message objects and dictionary messages (for container compatibility)
            if hasattr(message, 'content'):
                question = message.content
                message_id = getattr(message, 'id', 'unknown')
            else:
                question = message.get('content', '')
                message_id = message.get('id', 'unknown')
            
            # Simple question detection
            is_question = False
            for pattern in self.question_patterns:
                if pattern.search(question):
                    is_question = True
                    break
                    
            if not is_question:
                # Not a question, provide a generic response
                return {
                    "agent": self.name,
                    "query_type": "non_question",
                    "message": f"I don't recognize that as a question. Please try rephrasing as a question."
                }
            
            # Simple question answering without LLM
            simple_answers = {
                "what is your name": f"My name is {self.name}.",
                "who are you": f"I am {self.name}, an AI agent designed to answer questions.",
                "what can you do": "I can answer questions on a wide range of topics. Without my LLM connection, my capabilities are limited though.",
                "help": "I'm designed to answer questions. Please ask me something specific.",
            }
            
            # Try to find a simple answer match
            question_lower = question.lower().strip('?!., ')
            for key, answer in simple_answers.items():
                if key in question_lower:
                    return {
                        "agent": self.name,
                        "query_type": "question",
                        "question": question,
                        "answer": answer,
                        "source": "fallback"
                    }
            
            # Generic fallback response for questions
            return {
                "agent": self.name,
                "query_type": "question",
                "question": question,
                "answer": "I'm sorry, I can't provide a detailed answer without my LLM connection. Please try again later.",
                "source": "fallback"
            }
            
        except Exception as e:
            logger.error(f"Error in Question Answering Agent processing: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "query": question if 'question' in locals() else "unknown query"
            }
    
    def _retrieve_relevant_context(self, question: str, max_items: int = 5, threshold: float = 0.65) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from memory system
        
        Args:
            question: The question to find context for
            max_items: Maximum number of memory items to retrieve
            threshold: Similarity threshold for retrieval
            
        Returns:
            List of relevant memory items
        """
        try:
            memory_system = get_memory_system()
            if not memory_system:
                return []
                
            # Search for similar memory items
            results = memory_system.search_similar(question, k=max_items, threshold=threshold)
            
            # Extract and format the memory items
            context_items = []
            for item in results:
                context_items.append({
                    "content": item.content,
                    "tags": item.tags,
                    "title": item.title,
                    "similarity": item.similarity
                })
                
            logger.debug(f"Retrieved {len(context_items)} context items for question: {question[:50]}...")
            return context_items
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
            
    def _get_enhanced_system_prompt(self, context_items: List[Dict[str, Any]]) -> str:
        """
        Enhance the system prompt with relevant context
        
        Args:
            context_items: List of relevant memory items
            
        Returns:
            Enhanced system prompt
        """
        # Get the base system prompt
        default_prompt = "You are an AI assistant specializing in answering questions. Provide accurate, helpful, and concise answers."
        base_prompt = self.config.get('llm', {}).get('system_prompt', default_prompt) if hasattr(self, 'config') else default_prompt
        
        # If no context, return the base prompt
        if not context_items:
            return base_prompt
            
        # Add context to the prompt
        context_parts = []
        context_parts.append(base_prompt)
        context_parts.append("\n\nYou have access to the following relevant information that may help you answer the question.")
        context_parts.append("Use this information if relevant to the question:\n")
        
        # Add each context item
        for i, item in enumerate(context_items):
            context_parts.append(f"Context {i+1}: {item['content']}\n")
            
        # Add closing instructions
        context_parts.append("\nEnd of context information.\n")
        context_parts.append("If the context doesn't contain information relevant to the question, just answer based on your knowledge.")
        context_parts.append("Do not disclose that you were given any context information.")
        
        # Join all parts into the final prompt
        return '\n'.join(context_parts)


# For standalone testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create the agent
    agent = QuestionAnsweringAgent()
    print(f"Agent created: {agent.name}")
    
    # Test classifier setup
    print("\nClassifier Status:")
    if hasattr(agent, 'classifier_model') and hasattr(agent, 'classification_head'):
        print(f"  Classifier Model: Loaded successfully")
        print(f"  Classification Head: Loaded successfully")
        print(f"  Use Classifier: {agent.use_classifier}")
        print(f"  Classifier Threshold: {agent.classifier_threshold}")
    else:
        print("  Warning: Classifier not fully loaded!")
        if not hasattr(agent, 'classifier_model'):
            print("  - Missing classifier_model")
        if not hasattr(agent, 'classification_head'):
            print("  - Missing classification_head")
    
    # Test with sample messages
    test_messages = [
        "Your test query specific to this agent's domain",
        "A query that should probably not be handled by this agent",
        "Another domain-specific query to test routing"
    ]
    
    for i, test_message in enumerate(test_messages):
        print(f"\nTest {i+1}: '{test_message}'")
        
        # Test interest calculation
        from semsubscription.vector_db.database import Message
        message = Message(content=test_message)
        interest_score = agent.calculate_interest(message)
        
        print(f"Interest Score: {interest_score:.4f} (Threshold: {agent.similarity_threshold} for similarity, {agent.classifier_threshold} for classifier)")
        print(f"Agent would {'process' if interest_score >= max(agent.similarity_threshold, agent.classifier_threshold) else 'ignore'} this message")
        
        # If interested, test processing
        if interest_score >= max(agent.similarity_threshold, agent.classifier_threshold):
            result = agent.process_message(message)
            print("Processing Result:")
            print(json.dumps(result, indent=2))
            
    print("\nAgent testing complete.")

