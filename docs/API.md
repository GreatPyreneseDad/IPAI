# IPAI API Documentation

## Overview

The IPAI API provides RESTful endpoints for interacting with the Individually Programmed AI system. All endpoints return JSON responses and require authentication unless otherwise noted.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

IPAI uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### Authentication

#### Register User
```http
POST /auth/register
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "securepassword123",
  "full_name": "John Doe"
}
```

**Response:**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "is_active": true,
  "is_verified": false,
  "role": "user",
  "created_at": "2023-12-01T00:00:00Z",
  "current_coherence_score": 1.0,
  "coherence_level": "moderate"
}
```

#### Login
```http
POST /auth/login
```

**Request Body (form-data):**
```
username: johndoe
password: securepassword123
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Coherence Management

#### Get Current Coherence
```http
GET /coherence/current
```

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "psi": 0.85,
  "rho": 0.72,
  "q": 0.91,
  "f": 0.68,
  "coherence_score": 0.79,
  "level": "moderate",
  "timestamp": "2023-12-01T12:00:00Z",
  "safety_metrics": {
    "safety_score": 0.92,
    "howlround_risk": 0.15,
    "intervention_needed": false
  },
  "pressure_metrics": {
    "pressure_score": 0.23,
    "adaptation_rate": 0.87
  }
}
```

#### Submit Assessment
```http
POST /coherence/assessment/submit
```

**Request Body:**
```json
{
  "assessment_type": "initial",
  "responses": {
    "question_1": "answer_1",
    "question_2": "answer_2"
  }
}
```

#### Get Coherence History
```http
GET /coherence/history?days=30
```

**Query Parameters:**
- `days` (optional): Number of days to retrieve (default: 30)

### LLM Chat

#### Send Chat Message
```http
POST /llm/chat
```

**Request Body:**
```json
{
  "message": "Hello, how can I improve my coherence?",
  "context": [],
  "provider": "ollama",
  "temperature": 0.7,
  "max_tokens": 2000
}
```

**Response:**
```json
{
  "message": "To improve your coherence, I recommend focusing on...",
  "coherence_impact": 0.02,
  "safety_score": 0.95,
  "metadata": {
    "triadic_components": {},
    "current_coherence": 0.81,
    "coherence_state": "moderate",
    "relationship_quality": 0.88
  }
}
```

#### Stream Chat (WebSocket)
```
WS /llm/chat/stream
```

**Authentication:**
Send token as query parameter: `?token=<your-jwt-token>`

**Message Format:**
```json
{
  "type": "message",
  "message": "Your question here",
  "provider": "ollama",
  "temperature": 0.7
}
```

**Response Events:**
- `chunk`: Streaming response chunks
- `complete`: Final response with coherence data
- `error`: Error message

### Configuration

#### List LLM Providers
```http
GET /llm/models
```

**Response:**
```json
{
  "providers": {
    "Ollama": {
      "provider_type": "ollama",
      "models": ["llama3.2", "codellama"],
      "active": true
    }
  },
  "active_provider": "ollama"
}
```

#### Set Active Provider
```http
POST /llm/set-provider?provider_name=ollama
```

### User Management

#### Get Current User
```http
GET /users/me
```

#### Update Profile
```http
PUT /users/me
```

**Request Body:**
```json
{
  "full_name": "John Doe Updated",
  "bio": "AI researcher and enthusiast"
}
```

#### Get User Interactions
```http
GET /users/interactions?limit=10
```

### Analytics

#### Get Coherence Analytics
```http
GET /coherence/analytics
```

**Response:**
```json
{
  "trends": [],
  "insights": [
    "Your coherence has improved by 15% this month"
  ],
  "risk_factors": [],
  "growth_opportunities": [
    "Consider daily meditation practice"
  ]
}
```

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

Common HTTP status codes:
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

## Rate Limiting

Default rate limits:
- 60 requests per minute
- 1000 requests per hour

Rate limit headers:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1638360000
```

## Pagination

For endpoints returning lists:

**Query Parameters:**
- `skip`: Number of items to skip (default: 0)
- `limit`: Number of items to return (default: 50, max: 100)

**Response Headers:**
```
X-Total-Count: 150
X-Page-Count: 3
```

## WebSocket Events

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/llm/chat/stream?token=<token>');
```

### Message Types
- `ping`: Keep-alive
- `message`: Chat message
- `chunk`: Streaming response
- `complete`: End of stream
- `error`: Error occurred

## SDK Examples

### Python
```python
import requests

# Login
response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    data={"username": "johndoe", "password": "password123"}
)
tokens = response.json()

# Use API
headers = {"Authorization": f"Bearer {tokens['access_token']}"}
coherence = requests.get(
    "http://localhost:8000/api/v1/coherence/current",
    headers=headers
).json()
```

### JavaScript/TypeScript
```typescript
// Using the frontend service layer
import { llmService } from '@/services/llm.service';

const response = await llmService.chat({
  message: "Hello IPAI",
  temperature: 0.7
});
```

## Postman Collection

A Postman collection is available in `docs/IPAI.postman_collection.json` for easy API testing.