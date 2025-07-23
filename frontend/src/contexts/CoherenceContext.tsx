import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { CoherenceProfile } from '@/types'
import { coherenceService, CoherenceSnapshot } from '@/services/coherence.service'
import { useAuth } from './AuthContext'
import { useToast } from '@/components/ui/use-toast'

interface CoherenceContextType {
  currentCoherence: CoherenceSnapshot | null
  coherenceHistory: CoherenceProfile[]
  isLoading: boolean
  refreshCoherence: () => Promise<void>
  loadHistory: (days?: number) => Promise<void>
}

const CoherenceContext = createContext<CoherenceContextType | undefined>(undefined)

export const useCoherence = () => {
  const context = useContext(CoherenceContext)
  if (!context) {
    throw new Error('useCoherence must be used within a CoherenceProvider')
  }
  return context
}

interface CoherenceProviderProps {
  children: ReactNode
}

export const CoherenceProvider: React.FC<CoherenceProviderProps> = ({ children }) => {
  const [currentCoherence, setCurrentCoherence] = useState<CoherenceSnapshot | null>(null)
  const [coherenceHistory, setCoherenceHistory] = useState<CoherenceProfile[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const { user } = useAuth()
  const { toast } = useToast()
  
  const refreshCoherence = async () => {
    if (!user) return
    
    try {
      setIsLoading(true)
      const snapshot = await coherenceService.getCurrentCoherence()
      setCurrentCoherence(snapshot)
      
      // Check for safety alerts
      if (snapshot.safetyMetrics.interventionNeeded) {
        toast({
          title: 'Coherence Alert',
          description: 'Your coherence levels require attention. Consider taking a break or engaging in grounding activities.',
          variant: 'destructive',
        })
      }
    } catch (error) {
      console.error('Failed to refresh coherence:', error)
    } finally {
      setIsLoading(false)
    }
  }
  
  const loadHistory = async (days: number = 30) => {
    if (!user) return
    
    try {
      const history = await coherenceService.getCoherenceHistory(days)
      setCoherenceHistory(history)
    } catch (error) {
      console.error('Failed to load coherence history:', error)
    }
  }
  
  // Load coherence data when user changes
  useEffect(() => {
    if (user) {
      refreshCoherence()
      loadHistory()
    } else {
      setCurrentCoherence(null)
      setCoherenceHistory([])
    }
  }, [user])
  
  // Auto-refresh coherence every 5 minutes
  useEffect(() => {
    if (!user) return
    
    const interval = setInterval(() => {
      refreshCoherence()
    }, 5 * 60 * 1000)
    
    return () => clearInterval(interval)
  }, [user])
  
  return (
    <CoherenceContext.Provider
      value={{
        currentCoherence,
        coherenceHistory,
        isLoading,
        refreshCoherence,
        loadHistory,
      }}
    >
      {children}
    </CoherenceContext.Provider>
  )
}