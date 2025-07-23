import { api } from './api'
import { ChatMessage, ChatResponse, LLMProvider } from '@/types'
import { io, Socket } from 'socket.io-client'

export interface ChatRequest {
  message: string
  context?: ChatMessage[]
  provider?: string
  model?: string
  temperature?: number
  maxTokens?: number
  stream?: boolean
}

export interface AnalysisRequest {
  text: string
  analysisType: 'coherence' | 'sentiment' | 'themes' | 'summary'
  depth?: 'quick' | 'standard' | 'deep'
}

export interface CompletionRequest {
  prompt: string
  systemPrompt?: string
  examples?: Array<{ input: string; output: string }>
  provider?: string
  model?: string
  temperature?: number
  maxTokens?: number
}

class LLMService {
  private socket: Socket | null = null
  
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await api.post('/llm/chat', request)
    return response.data
  }
  
  async streamChat(
    request: ChatRequest,
    onChunk: (chunk: string) => void,
    onComplete: (response: any) => void,
    onError: (error: any) => void
  ): Promise<() => void> {
    // Get auth token
    const token = localStorage.getItem('accessToken')
    
    // Create WebSocket connection
    this.socket = io(import.meta.env.VITE_WS_URL || 'ws://localhost:8000', {
      transports: ['websocket'],
      auth: {
        token,
      },
    })
    
    // Handle connection
    this.socket.on('connect', () => {
      // Send chat request
      this.socket?.emit('chat', request)
    })
    
    // Handle chunks
    this.socket.on('chunk', (data) => {
      onChunk(data.content)
    })
    
    // Handle completion
    this.socket.on('complete', (data) => {
      onComplete(data)
      this.disconnect()
    })
    
    // Handle errors
    this.socket.on('error', (error) => {
      onError(error)
      this.disconnect()
    })
    
    // Return cleanup function
    return () => {
      this.disconnect()
    }
  }
  
  private disconnect() {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
  }
  
  async analyze(request: AnalysisRequest): Promise<any> {
    const response = await api.post('/llm/analyze', request)
    return response.data
  }
  
  async complete(request: CompletionRequest): Promise<{
    completion: string
    promptTokens: number
    completionTokens: number
    model: string
  }> {
    const response = await api.post('/llm/complete', request)
    return response.data
  }
  
  async getProviders(): Promise<LLMProvider[]> {
    const response = await api.get('/llm/models')
    return Object.entries(response.data.providers).map(([name, data]: any) => ({
      provider: data.provider_type,
      name,
      models: data.models,
      active: data.active,
      configured: data.active,
    }))
  }
  
  async setActiveProvider(providerName: string): Promise<void> {
    await api.post('/llm/set-provider', null, {
      params: { provider_name: providerName },
    })
  }
  
  async getConversationHistory(limit: number = 50): Promise<{
    history: ChatMessage[]
    totalMessages: number
    sessionStart: string
  }> {
    const response = await api.get('/llm/conversation-history', {
      params: { limit },
    })
    return response.data
  }
  
  async clearConversationHistory(): Promise<void> {
    await api.delete('/llm/conversation-history')
  }
}

export const llmService = new LLMService()