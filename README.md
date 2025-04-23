# Question_answering_agent Agent

## Overview

Detects and answers questions using an LLM

This agent is part of the Semantic Subscription System, an event-driven distributed message processing framework that uses semantic similarity to determine which agents should process which messages.

## System Architecture

### Semantic Subscription System

The Semantic Subscription System operates on the principle of "semantic subscription," where messages are routed to agents based on their semantic meaning rather than explicit routing rules. This is implemented using vector embeddings where:

1. Each message is converted to a vector representation (embedding)
2. Each agent has its own set of interest vectors from fine-tuned models
3. Agents "subscribe" to messages whose vectors are similar to their interest vectors

### Event-Driven Architecture

The system uses an event-driven architecture with a two-phase message processing approach:

1. **Interest Registration Phase**: Determines which agents are interested in a message
2. **Processing Phase**: Interested agents process the message and create responses

Events are published to a distributed event bus (Redis) and agents subscribe to specific event types.

### Containerization

Agents run in individual Docker containers, which are deployed from GitHub repositories. The containerization workflow:

1. Agent code is stored in GitHub repositories (like this one)
2. The container is built from the repository code
3. The container connects to the Redis event bus and registers with the core system
4. When deployed, the agent receives events and processes messages based on its interests

## Agent Implementation

### Key Files

- `agent.py`: Main agent implementation with interest determination and message processing logic
- `config.yaml`: Configuration including thresholds, polling settings, and training parameters
- `agent_base.py`: Base functionality for containerized operations
- `agent_container.py`: Entry point for the containerized agent
- `interest_model.py`: Handles message interest determination
- `examples.jsonl`: Training examples for the fine-tuned interest model
- `models/`: Directory containing the fine-tuned interest model (created during training)

### Agent Configuration

This agent is configured in `config.yaml` with the following key settings:

```yaml
# Basic information
agent:
  name: "Question_answering_agent Agent"
  package_name: "question_answering_agent"
  description: "Detects and answers questions using an LLM"
  domain: "general knowledge question answering"

# Interest model configuration
interest_model:
  threshold: 0.67                  # Similarity threshold
  clustering_method: "kmeans"      # Clustering algorithm
  num_clusters: 3                  # Number of clusters for examples
  dimension: 384                   # Embedding dimension

use_classifier: true              # Use fine-tuned classifier model
classifier_threshold: 0.5         # Threshold for classifier decisions
polling_interval: 1.0             # Fallback polling interval (seconds)
```

## Message Processing Flow

The agent processes messages through the following flow:

1. **Interest Determination**:
   - The agent receives a `message.created` event
   - It uses its fine-tuned model to calculate an interest score for the message
   - If the score exceeds the threshold, the agent marks interest in the message

2. **Message Processing**:
   - The agent receives a `message.interest.registered` event for messages it's interested in
   - It processes the message using domain-specific logic in the `process_message` method
   - The response is added back to the system as a new message

### Implementing the Agent Logic

The primary methods to implement in your agent are:

```python
def setup_interest_model(self):
    """Configure the agent's interest model by providing examples or loading a fine-tuned model"""
    # This is called during initialization
    pass

def is_interested(self, message: Message) -> bool:
    """Determine if the agent is interested in a message (optional override)"""
    # By default, uses the fine-tuned model's classifier
    # Return True if interested, False otherwise
    pass

def process_message(self, message: Message) -> Optional[Dict[str, Any]]:
    """Process a message that matches the agent's interests"""
    # Implement your domain-specific processing logic here
    # Return a dictionary with processing results or None
    return {"result": "Processed message: " + message.content}
```

## LLM Integration

This agent can leverage the system's LLM capabilities for processing messages. The system provides:

```python
from semsubscription.llm.completion import get_completion

def process_message(self, message: Message) -> Optional[Dict[str, Any]]:
    # Example of using the LLM in your agent
    response = get_completion(
        prompt=f"Process this message: {message.content}",
        model="gpt-4-1106-preview",  # or other available models
        temperature=0.7
    )
    return {"result": response}
```

## Fine-Tuned Model Training

The agent uses a fine-tuned model for interest determination. The training process:

1. Create examples in `examples.jsonl` with positive and negative instances
2. Run the fine-tuning process through the web portal or directly with the script
3. The model is saved to the `models/` directory and used for interest classification

Example format for `examples.jsonl`:
```json
{"text": "This is an example message the agent should be interested in", "label": 1}
{"text": "This is an example the agent should ignore", "label": 0}
```

## Testing and Debugging

### Local Testing

To test this agent locally:

```bash
# Test the agent directly
python agent.py

# Test using the framework
python ../../test_framework.py agent question_answering_agent

# Test classification model
python ../../test_framework.py classifier question_answering_agent
```

### Containerized Testing

The agent can also be tested in the containerized environment:

1. Deploy the agent through the web portal
2. Send test messages via the API
3. Check agent logs for debug information

```bash
# Send a test message via curl
curl -X POST "http://localhost:8888/api/messages/" \
  -H "Content-Type: application/json" \
  -d '{"content": "Your test message here"}'

# Check container logs
docker logs agent-question_answering_agent-container
```

## API Integration

The agent communicates with the core system through the Redis event bus. You can also interact with the agent through the REST API:

```
POST /api/messages/              # Create a new message
GET /api/messages/{message_id}   # Get a specific message
GET /api/messages/{message_id}/responses # Get responses to a message
```

See the full API documentation for more endpoints.

## Advanced Features

### Custom Embedding Models

The agent supports custom embedding models by extending the `embedding_engine.py` module. This allows for domain-specific embeddings that can better capture the semantic meaning of messages in your specific context.

### Interest Area Refinement

You can refine the agent's interest areas by:

1. Adding more diverse examples to `examples.jsonl`
2. Adjusting the `classifier_threshold` in `config.yaml`
3. Implementing custom logic in the `is_interested` method

## Deployment

This agent will be automatically built and deployed as a container when:  

1. The repository is connected to the system through the web portal
2. The agent is deployed via the agent deployment API
3. Changes are pushed to the main branch of this repository

The containerization process automatically handles environment setup, dependency installation, and connection to the event bus.
