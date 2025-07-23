export interface User {
  id: string
  email: string
  username: string
  fullName?: string
  bio?: string
  avatarUrl?: string
  role: UserRole
  isActive: boolean
  isVerified: boolean
  currentCoherenceScore: number
  coherenceLevel: CoherenceLevel
  walletAddress?: string
  sageBalance: number
  createdAt: string
  lastLogin?: string
}

export enum UserRole {
  USER = 'user',
  PREMIUM = 'premium',
  ADMIN = 'admin',
  MODERATOR = 'moderator'
}

export enum CoherenceLevel {
  CRITICAL = 'critical',
  LOW = 'low',
  MODERATE = 'moderate',
  HIGH = 'high',
  OPTIMAL = 'optimal'
}

export interface CoherenceProfile {
  psi: number  // Internal consistency
  rho: number  // Accumulated wisdom
  q: number    // Moral activation energy
  f: number    // Social belonging
  coherenceScore: number
  soulEcho: number
  level: CoherenceLevel
  calculatedAt: string
  riskFactors: Record<string, any>
  growthPotential: number
  stabilityIndex: number
}

export interface ChatMessage {
  id?: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
  coherenceImpact?: number
  metadata?: Record<string, any>
}

export interface ChatResponse {
  message: string
  coherenceImpact: number
  safetyScore: number
  metadata: {
    triadicComponents?: Record<string, any>
    currentCoherence: number
    coherenceState: string
    relationshipQuality: number
  }
}

export interface LLMProvider {
  provider: string
  name: string
  models: string[]
  active: boolean
  configured: boolean
}

export interface WalletProvider {
  name: string
  icon: string
  chainId: number
  rpcUrl?: string
}

export interface UserInteraction {
  id: number
  interactionType: 'chat' | 'assessment' | 'meditation' | 'journal' | 'analysis'
  inputText: string
  outputText: string
  coherenceBefore: number
  coherenceAfter: number
  coherenceDelta: number
  safetyScore: number
  createdAt: string
  blockHash?: string
}

export interface Assessment {
  id: number
  assessmentType: string
  version: string
  questions: any[]
  responses: any[]
  psiScore: number
  rhoScore: number
  qScore: number
  fScore: number
  completedAt: string
  completionRate: number
}

export interface Notification {
  id: number
  type: string
  title: string
  message: string
  isRead: boolean
  createdAt: string
  data?: Record<string, any>
}

export interface AuthTokens {
  accessToken: string
  refreshToken: string
  tokenType: string
  expiresIn: number
}