import os
import time
import json
import logging
import requests
import yaml

# Import the specific agent class based on config
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    agent_class_name = config.get('agent', {}).get('class_name', 'Test')
    package_name = config.get('agent', {}).get('package_name', 'test_agent')
except Exception as e:
    agent_class_name = 'Test' 
    package_name = 'test_agent'
    print(f"Warning: Could not load config.yaml: {e}")

# Dynamic import of the agent class
module_name = f"agent"
try:
    import importlib
    agent_module = importlib.import_module(module_name)
    AgentClass = getattr(agent_module, f"{agent_class_name}Agent")
    print(f"Successfully imported {agent_class_name}Agent from {module_name}")
except Exception as e:
    # Fallback to base implementation
    from agent import BaseAgent as AgentClass
    print(f"Warning: Using fallback BaseAgent: {e}")

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables
AGENT_ID = os.environ.get("AGENT_ID", "unknown")
AGENT_NAME = os.environ.get("AGENT_NAME", "Unknown Agent")
CORE_API_URL = os.environ.get("CORE_API_URL", "http://host.docker.internal:8000")

def register_with_core_system():
    """Register this agent with the core system"""
    try:
        response = requests.post(
            f"{CORE_API_URL}/api/agents/register",
            json={
                "agent_id": AGENT_ID,
                "name": AGENT_NAME,
                "container_id": os.environ.get("HOSTNAME", "unknown"),
                "status": "running"
            }
        )
        if response.status_code == 200:
            logger.info(f"Agent registered successfully with core system")
            return True
        else:
            logger.error(f"Failed to register with core system: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error registering with core system: {e}")
        return False

def process_messages():
    """Poll for messages to process"""
    try:
        # Check for pending messages using the API for containerized agents
        response = requests.get(
            f"{CORE_API_URL}/api/messages/pending",
            params={"agent_id": AGENT_ID}
        )
        
        if response.status_code == 200:
            messages = response.json()
            if messages:
                logger.info(f"Received {len(messages)} messages to process")
                for message in messages:
                    process_message(message)
            # Even if no messages, we register with the event-driven system
            # to make sure we're in the list of available agents
            subscribe_to_events()
        else:
            logger.error(f"Failed to get pending messages: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error checking for messages: {e}")

def subscribe_to_events():
    """Subscribe agent to the event-driven system"""
    try:
        # Register for new message events - this endpoint may not exist yet 
        # as we're transitioning to the event-driven architecture
        response = requests.post(
            f"{CORE_API_URL}/api/agents/subscribe",
            json={
                "agent_id": AGENT_ID,
                "name": AGENT_NAME,
                "events": ["message.new"],
                "callback_url": f"{CORE_API_URL}/api/messages/process"
            }
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully subscribed to events")
            return True
        elif response.status_code == 404:
            # This is expected during the transition to event-driven architecture
            # Fall back to polling mode silently without warning logs
            logger.debug(f"Event subscription not implemented yet, using polling mode")
            return False  
        else:
            logger.warning(f"Failed to subscribe to events: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        # Simply log at debug level since this endpoint may not exist yet
        logger.debug(f"Error subscribing to events: {e}")
        return False

def process_message(message):
    """Process a single message"""
    try:
        # Initialize the agent
        agent = AgentClass()
        
        # Calculate interest score
        interest_score = agent.calculate_interest(message)
        logger.info(f"Interest score for message {message['id']}: {interest_score}")
        
        # First phase: Register interest with the event-driven system
        # This corresponds to the AgentInterestService in the core system
        interest_response = requests.post(
            f"{CORE_API_URL}/api/messages/{message['id']}/interest",
            json={
                "agent_id": AGENT_ID,
                "name": AGENT_NAME,
                "score": interest_score
            }
        )
        
        # Second phase: If interest score exceeds threshold, process the message
        # This corresponds to the MessageProcessingService in the core system
        if interest_score >= agent.classifier_threshold:
            # Process the message and get the result
            result = agent.process_message(message)
            
            if result:
                # Include agent information in the result
                if isinstance(result, dict) and 'agent_id' not in result:
                    result['agent_id'] = AGENT_ID
                
                # Submit processing result to core system
                process_response = requests.post(
                    f"{CORE_API_URL}/api/messages/{message['id']}/process",
                    json={
                        "agent_id": AGENT_ID,
                        "result": result
                    }
                )
                
                if process_response.status_code == 200:
                    logger.info(f"Successfully processed message {message['id']}")
                else:
                    logger.warning(f"Error submitting processing result: {process_response.status_code} - {process_response.text}")
            else:
                logger.info(f"Agent returned no result for message {message['id']}")
        else:
            logger.info(f"Interest score {interest_score} below threshold {agent.classifier_threshold}, skipping processing")
            
    except Exception as e:
        logger.error(f"Error processing message {message.get('id', 'unknown')}: {e}")

def main():
    logger.info(f"Starting agent container for {AGENT_NAME} (ID: {AGENT_ID})")
    
    # Register with core system
    if not register_with_core_system():
        logger.warning("Continuing without registration...")
    
    # Main processing loop
    try:
        while True:
            process_messages()
            time.sleep(5)  # Poll every 5 seconds
    except KeyboardInterrupt:
        logger.info("Shutting down agent container")

if __name__ == "__main__":
    main()
