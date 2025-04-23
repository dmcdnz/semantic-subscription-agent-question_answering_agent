#!/usr/bin/env python

"""
Custom Interest Model Implementation for {agent_name}

Extends the base interest model functionality with domain-specific features.
"""

import logging
from typing import List, Optional

# Import the InterestModel directly using our updated import structure
from semsubscription.vector_db.embedding import InterestModel

logger = logging.getLogger(__name__)


class CustomInterestModel(InterestModel):
    """Custom interest model with domain-specific enhancements."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add any custom initialization here
    
    def is_interested(self, text: str) -> bool:
        """Determine if the agent is interested in the given text.
        
        You can override this method to add custom logic beyond the base
        similarity calculation. For example, you might add keyword matching,
        regex patterns, or other domain-specific heuristics.
        
        Args:
            text: The text to evaluate
            
        Returns:
            Boolean indicating interest
        """
        # First check using the base similarity method
        base_interest = super().is_interested(text)
        
        # You can add custom logic here to supplement the base calculation
        # For example:
        # if any(keyword in text.lower() for keyword in ["weather", "temperature", "forecast"]):
        #     return True
        
        return base_interest
    
    def calculate_similarity(self, text: str) -> float:
        """Calculate similarity between the text and interest vectors.
        
        You can override this method to implement custom similarity calculations.
        
        Args:
            text: The text to calculate similarity for
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Use the base calculation or implement custom logic
        return super().calculate_similarity(text)
