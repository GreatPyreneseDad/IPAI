# IPAI Architecture Documentation

## System Overview

IPAI (Individually Programmed AI) is a comprehensive system designed to preserve and amplify authentic human consciousness through mathematical frameworks. The architecture emphasizes modularity, scalability, and data sovereignty.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Dashboard   │  │     Chat     │  │   Assessment     │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTPS/WSS
┌───────────────────────────┴─────────────────────────────────┐
│                    API Gateway (FastAPI)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │    Auth     │  │  Coherence   │  │      LLM         │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                      Core Services                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Coherence  │  │    Safety    │  │   Blockchain     │   │
│  │   Engine    │  │   Systems    │  │   Integration    │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                    Data Layer                                │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ PostgreSQL  │  │    Redis     │  │     Ollama       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Frontend Layer

#### React Application
- **Framework**: React 18 with TypeScript
- **State Management**: React Context + Zustand
- **Routing**: React Router v6
- **UI Components**: Custom components with Radix UI
- **Styling**: TailwindCSS
- **Data Fetching**: React Query

```typescript
// Component structure
src/
├── components/       # Reusable UI components
├── pages/           # Route-based pages
├── services/        # API integration layer
├── contexts/        # Global state management
├── hooks/           # Custom React hooks
└── utils/           # Utility functions
```

### 2. API Layer

#### FastAPI Application
- **Framework**: FastAPI with async support
- **Authentication**: JWT with refresh tokens
- **Validation**: Pydantic models
- **Documentation**: Auto-generated OpenAPI/Swagger
- **Middleware**: CORS, rate limiting, security headers

```python
# API structure
src/api/
├── v1/              # API version 1 endpoints
├── middleware/      # Custom middleware
├── dependencies/    # Dependency injection
└── exceptions/      # Exception handlers
```

#### Key Endpoints
- `/auth/*` - Authentication and user management
- `/coherence/*` - Coherence tracking and analysis
- `/llm/*` - LLM interactions with coherence
- `/config/*` - System configuration

### 3. Core Services

#### Coherence Engine
Mathematical implementation of Grounded Coherence Theory (GCT):

```python
class EnhancedGCTCalculator:
    """
    Implements the Soul Echo equation: Ψ·ρ·q·f
    - Ψ (psi): Internal consistency
    - ρ (rho): Accumulated wisdom
    - q: Moral activation energy
    - f: Social belonging
    """
```

#### Safety Systems
- **Howlround Detector**: Prevents feedback loops
- **Coherence Tracker**: Monitors psychological stability
- **Intervention System**: Automatic safety measures

```python
class HowlroundDetector:
    """
    Detects and prevents:
    - Recursive feedback loops
    - Amplification patterns
    - Ghost resonances
    """
```

#### Blockchain Integration
- **Personal Chains**: Individual blockchain for each user
- **Proof of Coherence**: Custom consensus mechanism
- **SAGE Tokens**: Rewards for coherence maintenance

### 4. Data Layer

#### PostgreSQL Database
Primary data storage with the following schema:

```sql
-- Core tables
users                 # User accounts and profiles
coherence_profiles    # GCT calculations
user_interactions     # Chat and interaction history
assessments          # Psychological assessments
inferences           # Verified inferences
```

#### Redis Cache
- Session management
- Real-time coherence data
- Rate limiting counters
- WebSocket connections

#### Ollama Integration
- Local LLM processing
- Privacy-preserving AI
- Multiple model support

## Data Flow

### 1. User Interaction Flow
```
User Input → Frontend → API Gateway → Auth Check → 
Coherence Engine → LLM Processing → Safety Check → 
Response Generation → Coherence Update → Frontend
```

### 2. Coherence Tracking Flow
```
Interaction → Coherence Calculator → GCT Analysis → 
Safety Evaluation → Blockchain Recording → 
Database Storage → Real-time Updates
```

### 3. Assessment Flow
```
Assessment Start → Question Generation → User Responses → 
Scoring Algorithm → Parameter Calibration → 
Profile Update → Blockchain Verification
```

## Security Architecture

### Authentication & Authorization
- **JWT Tokens**: Short-lived access tokens (30 min)
- **Refresh Tokens**: Long-lived refresh tokens (7 days)
- **Role-Based Access**: User, Premium, Admin roles
- **API Key Management**: For external integrations

### Data Security
- **Encryption at Rest**: AES-256 for sensitive data
- **Encryption in Transit**: TLS 1.3 for all connections
- **Data Sovereignty**: User owns and controls all data
- **Local Processing**: LLM runs on-premises

### Network Security
- **Rate Limiting**: Per-user and global limits
- **CORS Policy**: Strict origin validation
- **Input Validation**: Comprehensive sanitization
- **SQL Injection Prevention**: Parameterized queries

## Scalability Design

### Horizontal Scaling
- **Stateless API**: Can scale to multiple instances
- **Load Balancing**: Nginx or cloud load balancers
- **Database Pooling**: Connection pool management
- **Cache Layer**: Redis for shared state

### Vertical Scaling
- **Worker Processes**: Configurable worker count
- **Async Processing**: Non-blocking I/O
- **Resource Limits**: Container resource constraints
- **Auto-scaling**: Based on CPU/memory metrics

## Integration Points

### LLM Providers
- **Ollama**: Primary local LLM
- **OpenAI**: Optional cloud provider
- **Anthropic**: Optional cloud provider
- **Custom Models**: Pluggable architecture

### Blockchain Networks
- **Ethereum**: Identity NFTs
- **Polygon**: SAGE token transactions
- **Personal Chains**: User-specific blockchains

### External Services
- **Email**: SMTP integration
- **Webhooks**: Event notifications
- **Analytics**: Optional telemetry

## Performance Optimization

### Caching Strategy
- **Redis**: Hot data and sessions
- **PostgreSQL**: Query result caching
- **Frontend**: React Query caching
- **CDN**: Static asset delivery

### Database Optimization
- **Indexes**: Strategic index placement
- **Partitioning**: Time-based partitioning
- **Connection Pooling**: Efficient connections
- **Query Optimization**: EXPLAIN analysis

### Async Processing
- **Background Tasks**: Celery integration
- **WebSockets**: Real-time updates
- **Event Streaming**: Server-sent events
- **Batch Processing**: Bulk operations

## Monitoring & Observability

### Metrics Collection
- **Prometheus**: Time-series metrics
- **Grafana**: Visualization dashboards
- **Custom Metrics**: Business KPIs

### Logging
- **Structured Logging**: JSON format
- **Log Aggregation**: ELK stack compatible
- **Error Tracking**: Sentry integration
- **Audit Trails**: Security events

### Health Checks
- **Liveness Probes**: Container health
- **Readiness Probes**: Service availability
- **Dependency Checks**: External services

## Development Workflow

### Local Development
```bash
# Backend
uvicorn src.api.main:app --reload

# Frontend
npm run dev

# Full stack
docker-compose up
```

### Testing Strategy
- **Unit Tests**: Component isolation
- **Integration Tests**: API endpoints
- **E2E Tests**: User workflows
- **Performance Tests**: Load testing

### CI/CD Pipeline
- **Build**: Docker images
- **Test**: Automated test suite
- **Deploy**: Rolling updates
- **Monitor**: Post-deployment checks

## Future Architecture Considerations

### Microservices Migration
- Coherence Service
- LLM Service
- Blockchain Service
- User Service

### Event-Driven Architecture
- Event sourcing for coherence
- CQRS for read/write separation
- Message queuing for async tasks

### Multi-Region Deployment
- Geographic distribution
- Data residency compliance
- Edge computing for LLM

### Advanced Features
- Federated learning
- Homomorphic encryption
- Zero-knowledge proofs
- Decentralized storage