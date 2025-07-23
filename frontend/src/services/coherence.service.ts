import { api } from './api'
import { CoherenceProfile, Assessment } from '@/types'

export interface CoherenceSnapshot {
  psi: number
  rho: number
  q: number
  f: number
  coherenceScore: number
  level: string
  timestamp: string
  safetyMetrics: {
    safetyScore: number
    howlroundRisk: number
    interventionNeeded: boolean
  }
  pressureMetrics: {
    pressureScore: number
    adaptationRate: number
  }
}

export interface AssessmentRequest {
  assessmentType: 'initial' | 'periodic' | 'comprehensive'
  responses: Record<string, any>
}

export interface CalibrationRequest {
  targetPsi?: number
  targetRho?: number
  targetQ?: number
  targetF?: number
}

class CoherenceService {
  async getCurrentCoherence(): Promise<CoherenceSnapshot> {
    const response = await api.get('/coherence/current')
    return response.data
  }
  
  async getCoherenceHistory(days: number = 30): Promise<CoherenceProfile[]> {
    const response = await api.get('/coherence/history', {
      params: { days }
    })
    return response.data
  }
  
  async startAssessment(type: string = 'initial'): Promise<{ questions: any[] }> {
    const response = await api.post('/coherence/assessment/start', { type })
    return response.data
  }
  
  async submitAssessment(data: AssessmentRequest): Promise<Assessment> {
    const response = await api.post('/coherence/assessment/submit', data)
    return response.data
  }
  
  async getAssessments(): Promise<Assessment[]> {
    const response = await api.get('/coherence/assessments')
    return response.data
  }
  
  async calibrateParameters(data: CalibrationRequest): Promise<CoherenceProfile> {
    const response = await api.post('/coherence/calibrate', data)
    return response.data
  }
  
  async getSafetyStatus(): Promise<{
    howlroundDetected: boolean
    resonancePatterns: any[]
    interventionActive: boolean
    recommendations: string[]
  }> {
    const response = await api.get('/coherence/safety')
    return response.data
  }
  
  async getCoherenceAnalytics(): Promise<{
    trends: any[]
    insights: string[]
    riskFactors: any[]
    growthOpportunities: string[]
  }> {
    const response = await api.get('/coherence/analytics')
    return response.data
  }
}

export const coherenceService = new CoherenceService()