#!/usr/bin/env python

"""
Base Agent implementation for containerized agents

This standalone version doesn't require the full semsubscription package.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for containerized agents that don't require the full semsubscription package
    """
    
    def __init__(self, agent_id=None, name=None, description=None, similarity_threshold=0.7):
        """
        Initialize the agent with its parameters
        """
        self.agent_id = agent_id or os.environ.get("AGENT_ID", "unknown")
        self.name = name or os.environ.get("AGENT_NAME", "Unknown Agent")
        self.description = description or "A containerized agent"
        self.similarity_threshold = similarity_threshold
        self.classifier_threshold = 0.6  # Default threshold for interest
        
        logger.info(f"Initialized agent: {self.name} (ID: {self.agent_id})")
    
    def calculate_interest(self, message):
        """
        Calculate agent interest in a message
        
        Args:
            message: Message to calculate interest for
            
        Returns:
            Float interest score between 0 and 1
        """
        # Simple implementation - override in subclasses
        # Just check for keywords in basic version
        content = message.get("content", "").lower()
        
        # Get domain keywords from config
        keywords = self.get_keywords()
        
        # Check for keyword matches
        for keyword in keywords:
            if keyword.lower() in content:
                return 0.8  # High interest for keyword match
        
        return 0.1  # Low default interest
    
    def get_keywords(self):
        """
        Get domain keywords from config
        """
        # Try to load from config.yaml
        try:
            with open("config.yaml", "r") as f:
                import yaml
                config = yaml.safe_load(f)
                return config.get("interest_model", {}).get("keywords", [])
        except Exception as e:
            logger.warning(f"Error loading keywords from config: {e}")
            return ["test", "example", "demo"]
    
    def process_message(self, message):
        """
        Process a message
        
        Args:
            message: Message to process
            
        Returns:
            Optional result dictionary with response
        """
        # Simple echo implementation - override in subclasses
        return {
            "agent": self.name,
            "response": f"Processed message from {self.name}",
            "input": message.get("content", "")
        }
    
    def __str__(self):
        return f"{self.name} ({self.agent_id})"
