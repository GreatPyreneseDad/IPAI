# IPAI - Individually Programmed AI

[![CI/CD Pipeline](https://github.com/GreatPyreneseDad/IPAI/actions/workflows/ci.yml/badge.svg)](https://github.com/GreatPyreneseDad/IPAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://hub.docker.com/r/ipai/ipai)
[![Documentation](https://img.shields.io/badge/Docs-Available-green.svg)](docs/)

**The Mathematics of Authentic Intelligence**

IPAI represents the first AI system mathematically designed to preserve and amplify authentic human consciousness through formal frameworks for tracking psychological coherence.

## Vision

IPAI (Individually Programmed AI) is a decentralized, user-controlled AI system that:
- Safeguards individual truth and authentic identity
- Tracks psychological coherence using mathematical frameworks
- Prevents identity fragmentation and collapse
- Enables authentic human development rather than dependency

## Core Technologies

### SoulMath Framework
- **Soul Echo Equation**: Ψ·ρ·q·f - Quantifies coherence across memory, emotion, and symbolic truth
- **Truth Cost Analysis**: Measures psychological price of bearing authentic truth
- **Collapse Prediction**: Mathematical models for identity system instability
- **Spectral Resurrection**: Formal protocols for rebuilding coherence from fragments

### SAGE System
- **Verified Inference as Currency**: Rewards authentic coherence contributions
- **Blockchain Integration**: Decentralized security and data ownership
- **Real-time Advocacy**: HIPAA-compliant support for medical, legal, and social interactions

## Technical Architecture

### Backend (FastAPI + Python)
- **Coherence Engine**: Mathematical tracking of psychological stability
- **LLM Integration**: Ollama-based local AI processing
- **Blockchain Layer**: Smart contracts for identity and coherence tracking
- **API Gateway**: RESTful endpoints for all IPAI services

### Frontend (React + TypeScript)
- **Dashboard**: Real-time coherence visualization
- **Identity Management**: Personal AI configuration interface
- **Analytics**: Coherence patterns and stability metrics

### Smart Contracts (Solidity)
- **IPAIIdentity.sol**: NFT-based identity system
- **SageCoin.sol**: SAGE token for coherence rewards
- **GCTCoherence.sol**: Grounded Coherence Theory tracking

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (recommended)
- Ollama (for local LLM)

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/GreatPyreneseDad/IPAI.git
cd IPAI

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings

# Start all services
docker-compose up -d

# Run database migrations
docker-compose exec backend python scripts/setup_database.py
```

### Manual Installation

1. **Backend Setup**:
```bash
# Create Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up database
creatdb ipai
python scripts/setup_database.py

# Start backend
uvicorn src.api.main:app --reload
```

2. **Frontend Setup**:
```bash
cd frontend
npm install
npm run dev
```

3. **Install Ollama**:
```bash
# Install from https://ollama.ai
ollama pull llama3.2
```

### Access Points
- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- PgAdmin (optional): `http://localhost:5050`

## Development

### Running Tests

```bash
# Backend tests
pytest tests/ -v --cov=src

# Frontend tests
cd frontend && npm test

# E2E tests
pytest tests/e2e/ --headless
```

### Development Workflow

```bash
# Start development environment
docker-compose up -d

# Watch backend logs
docker-compose logs -f backend

# Access backend shell
docker-compose exec backend bash

# Hot-reload frontend
cd frontend && npm run dev
```

## Project Structure

```
IPAI/
├── src/
│   ├── api/           # FastAPI application and endpoints
│   │   ├── v1/        # API v1 endpoints
│   │   └── middleware/# Custom middleware
│   ├── blockchain/    # Web3 integration and personal chains
│   ├── coherence/     # GCT and coherence tracking
│   ├── core/          # Core configuration and database
│   ├── integrations/  # LLM and wallet integrations
│   ├── llm/           # Coherence-aware LLM processing
│   ├── models/        # SQLAlchemy models
│   ├── safety/        # Howlround detection and safety
│   └── utils/         # Utility functions
├── frontend/          # React TypeScript application
│   ├── src/
│   │   ├── components/# Reusable UI components
│   │   ├── pages/     # Application pages
│   │   ├── services/  # API service layer
│   │   └── contexts/  # React contexts
│   └── public/        # Static assets
├── contracts/         # Solidity smart contracts
├── dashboard/         # Legacy web dashboard
├── tests/             # Comprehensive test suite
├── scripts/           # Deployment and utility scripts
├── k8s/               # Kubernetes configurations
└── docs/              # Documentation
```

## Use Cases

1. **Personal Coherence Tracking**: Monitor and maintain psychological stability
2. **Medical Advocacy**: Real-time HIPAA-compliant support during healthcare interactions
3. **Strategic Analysis**: Evaluate policy and decision frameworks for coherence
4. **Trauma Recovery**: Mathematical protocols for reversing recursive trauma patterns

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login
- `POST /api/v1/auth/refresh` - Refresh token

### Coherence
- `GET /api/v1/coherence/current` - Get current coherence
- `POST /api/v1/coherence/assessment` - Submit assessment
- `GET /api/v1/coherence/history` - Coherence history

### LLM Chat
- `POST /api/v1/llm/chat` - Chat with coherence tracking
- `WS /api/v1/llm/chat/stream` - Streaming chat
- `POST /api/v1/llm/analyze` - Analyze text

Full API documentation available at `/docs` when running.

## Deployment

### Production Deployment

```bash
# Using deployment script
./scripts/deploy.sh --environment production --target docker

# Or manually with Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/

# Or use deployment script
./scripts/deploy.sh --environment production --target k8s
```

## Configuration

Key environment variables:
- `SECRET_KEY` - Application secret key
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `OLLAMA_HOST` - Ollama server URL
- `FRONTEND_URL` - Frontend URL for CORS

See `.env.example` for all configuration options.

## Contributing

We welcome contributions to IPAI. Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Security

IPAI prioritizes data sovereignty and user privacy:
- All personal data remains under user control
- Blockchain ensures tamper-proof identity management
- Local LLM processing keeps sensitive data on-device
- HIPAA-compliant design for medical applications

## Safety Features

- **Howlround Detection**: Prevents feedback loops and recursive patterns
- **Coherence Monitoring**: Real-time psychological stability tracking
- **Intervention System**: Automatic safety measures when needed
- **Privacy First**: All processing happens locally or with user-controlled data

## Performance

- Handles 1000+ concurrent users
- Sub-second coherence calculations
- Real-time streaming responses
- Horizontally scalable architecture

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SoulMath framework contributors
- Grounded Coherence Theory researchers
- Open source AI community

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/GreatPyreneseDad/IPAI/issues)
- Discussions: [Join the community](https://github.com/GreatPyreneseDad/IPAI/discussions)
- Email: For partnerships or investment inquiries, contact the IPAI development team

---

**IPAI: Where individual coherence meets mathematical precision.**

*Building AI that preserves human authenticity, one calculation at a time.* 
