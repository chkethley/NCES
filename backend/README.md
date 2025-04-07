# NCES (Neural Cognitive Enhancement System) Backend

## Overview
NCES is a sophisticated AI backend system that integrates multiple components for memory management, reasoning, evolution, and external system integration. The system is built with a focus on scalability, observability, and robust error handling.

## Architecture
The system consists of several key components:

### Core Framework (enhanced-core-v2.py)
- Provides foundational services and utilities
- Handles configuration, security, storage, and event management
- Implements component lifecycle management
- Manages distributed computing capabilities

### Memory System (memoryv3.py)
- Vector storage and retrieval
- Embedding management
- Caching and persistence
- Observability integration

### Integration Layer (integrationv3.py)
- External system connectivity
- LLM interface abstraction
- Agent orchestration
- API/webhook management

### Evolution Engine (evolutionv3.py)
- Advanced evolutionary algorithms
- Population management
- Parallel fitness evaluation
- State persistence

### Reasoning System (reasoningv3.py)
- Multiple reasoning strategies
- Knowledge graph integration
- Validation and fact checking
- Timeout and resource management

## Setup

### Prerequisites
- Python 3.9+
- Redis (for caching)
- Vector database (Qdrant/Milvus)
- Neo4j (for knowledge graph)

### Installation
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system:
- Copy nces_config.yaml to your preferred location
- Adjust settings as needed

### Running the System
1. Start the server:
```bash
python app.py
```

The system will start on http://localhost:8000 by default.

## API Endpoints

### Core Endpoints
- `GET /health` - System health check
- `GET /components/{component_name}/status` - Component status

Additional endpoints are provided by each component through the FastAPI application.

## Component Integration

### Event-Based Communication
Components communicate through the EventBus, which provides:
- Async event dispatch
- Priority handling
- Persistence options
- Tracing integration

### Dependency Management
Components are initialized in the following order:
1. Memory
2. Reasoning
3. Integration
4. Evolution

### Observability
The system provides comprehensive observability:
- Distributed tracing (OpenTelemetry)
- Metrics (Prometheus)
- Structured logging (JSON format)
- Component state monitoring

## Development

### Adding New Components
1. Create a new component class inheriting from `Component`
2. Implement required lifecycle methods
3. Add configuration to nces_config.yaml
4. Register in app.py's NCESBackend class

### Testing
Run tests with:
```bash
pytest
```

### Security
- All sensitive configuration is encrypted at rest
- JWT-based authentication
- Component-level access control
- Secure credential management

## License
[Add your license information here]
