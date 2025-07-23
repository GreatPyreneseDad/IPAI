# IPAI Frontend

React-based frontend for the Individually Programmed AI system.

## Tech Stack

- React 18 with TypeScript
- Vite for build tooling
- TailwindCSS for styling
- React Query for data fetching
- React Router for navigation
- Chart.js for data visualization
- Socket.io for real-time features
- Radix UI for accessible components

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Copy environment variables:
```bash
cp .env.example .env
```

3. Start development server:
```bash
npm run dev
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Project Structure

```
src/
├── components/     # Reusable UI components
├── contexts/       # React contexts (Auth, Coherence)
├── pages/          # Page components
├── services/       # API service layer
├── types/          # TypeScript type definitions
├── lib/            # Utility functions
├── hooks/          # Custom React hooks
├── store/          # State management
└── styles/         # Global styles
```

## Key Features

- Real-time coherence tracking
- AI-powered chat interface
- Psychological assessments
- Analytics dashboard
- Blockchain integration
- Multi-LLM provider support

## Environment Variables

- `VITE_API_URL` - Backend API URL (default: http://localhost:8000/api/v1)
- `VITE_WS_URL` - WebSocket URL (default: ws://localhost:8000)