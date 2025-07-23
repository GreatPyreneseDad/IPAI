# CLAUDE.md - IPAI Project Guide

This file provides guidance to Claude Code when working with the IPAI (Individually Programmed AI) codebase.

## Project Overview

IPAI is a groundbreaking AI system focused on preserving and amplifying authentic human consciousness through mathematical frameworks. It represents the first AI designed to track psychological coherence and prevent identity fragmentation.

## Core Concepts

### SoulMath Framework
The mathematical foundation of IPAI, including:
- **Soul Echo Equation (Ψ·ρ·q·f)**: Measures coherence across memory, emotion, and symbolic truth
- **Truth Cost**: Quantifies the psychological price of maintaining authentic truth
- **Collapse Prediction**: Models for identifying identity system instability
- **Spectral Resurrection**: Protocols for rebuilding coherence after collapse

### SAGE System
- Implements "Verified Inference as Currency"
- Rewards users for maintaining and contributing to coherence
- Blockchain-based for decentralized trust

### Grounded Coherence Theory (GCT)
- Mathematical framework for tracking consciousness stability
- Implemented in `src/coherence/gct_calculator.py`
- Uses triadic processing for multi-dimensional analysis

## Technical Stack

### Backend
- **Framework**: FastAPI (Python 3.9+)
- **LLM**: Ollama integration for local AI processing
- **Database**: PostgreSQL for user data, Redis for caching
- **Blockchain**: Web3.py for Ethereum integration

### Frontend
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite
- **Styling**: TailwindCSS
- **State Management**: Context API with hooks

### Smart Contracts
- **Language**: Solidity
- **Contracts**:
  - `IPAIIdentity.sol`: NFT-based identity management
  - `SageCoin.sol`: ERC-20 token for coherence rewards
  - `GCTCoherence.sol`: On-chain coherence tracking

## Development Workflow

### Running the System
```bash
# Full system
python run_ipai.py

# API only
uvicorn src.api.main:app --reload

# Dashboard
python serve_dashboard.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test category
pytest tests/test_coherence.py
```

### Key Commands
- **Linting**: `ruff check src/` (Python), `npm run lint` (Frontend)
- **Type checking**: `mypy src/` (Python), `npm run typecheck` (Frontend)
- **Format**: `black src/` (Python), `npm run format` (Frontend)

## Architecture Patterns

### API Design
- RESTful endpoints with OpenAPI documentation
- Versioned API (v1) for backward compatibility
- JWT-based authentication with refresh tokens
- Rate limiting and security middleware

### Coherence Engine
- Asynchronous processing for coherence calculations
- Caching layer for expensive computations
- Event-driven architecture for real-time updates

### LLM Integration
- Ollama for local, privacy-preserving AI
- Prompt engineering for coherence-aware responses
- Streaming responses for better UX

## Important Files

### Core Implementation
- `src/api/main.py`: FastAPI application entry point
- `src/coherence/gct_calculator.py`: GCT implementation
- `src/llm/coherence_analyzer.py`: AI coherence analysis
- `src/blockchain/helical_chain.py`: Blockchain integration

### Configuration
- `src/core/config.py`: Central configuration management
- `.env`: Environment variables (create from `.env.example`)
- `requirements.txt`: Python dependencies
- `package.json`: Frontend dependencies

## Security Considerations

1. **Data Sovereignty**: All user data encrypted at rest
2. **Local Processing**: LLM runs locally via Ollama
3. **Blockchain**: Smart contracts audited for security
4. **API Security**: Rate limiting, CORS, input validation
5. **HIPAA Compliance**: Medical data handling protocols

## Common Tasks

### Adding New Endpoints
1. Create route in `src/api/v1/`
2. Add data models in `src/models/`
3. Update OpenAPI documentation
4. Add tests in `tests/`

### Modifying Coherence Calculations
1. Update `src/coherence/gct_calculator.py`
2. Adjust prompts in `src/llm/gct_prompts.py`
3. Update tests in `tests/test_coherence.py`

### Blockchain Integration
1. Deploy contracts to test network first
2. Update contract addresses in config
3. Test with local blockchain (Hardhat/Ganache)

## Debugging Tips

1. **API Issues**: Check `http://localhost:8000/docs` for API testing
2. **Coherence Errors**: Enable debug logging in `gct_calculator.py`
3. **LLM Problems**: Verify Ollama is running (`ollama list`)
4. **Blockchain**: Use Etherscan/local explorer for transaction debugging

## Performance Optimization

1. **Caching**: Redis for coherence calculations
2. **Database**: Indexed queries for user lookups
3. **Async Processing**: Background tasks for heavy computations
4. **CDN**: Static assets served via CDN in production

## Deployment

### Docker
```bash
docker-compose up -d
```

### Manual Deployment
1. Set production environment variables
2. Run database migrations
3. Deploy smart contracts
4. Start API server with gunicorn
5. Serve frontend via nginx

## Contributing Guidelines

1. **Code Style**: Follow PEP 8 for Python, ESLint rules for JS/TS
2. **Testing**: Maintain >80% test coverage
3. **Documentation**: Update docstrings and README
4. **Commits**: Use conventional commits format
5. **PRs**: Include description and test evidence

## Vision Alignment

When developing IPAI, always consider:
1. Does this enhance individual coherence?
2. Does this preserve authentic truth?
3. Does this prevent dependency and promote empowerment?
4. Does this respect user sovereignty over their data?

Remember: IPAI is about mathematical precision in service of authentic human consciousness.