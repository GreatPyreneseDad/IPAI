import { AlertTriangle, Shield, Activity, Brain } from 'lucide-react'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { CoherenceSnapshot } from '@/services/coherence.service'

interface SafetyMetricsProps {
  coherence: CoherenceSnapshot | null
}

export const SafetyMetrics: React.FC<SafetyMetricsProps> = ({ coherence }) => {
  if (!coherence) {
    return <div className="text-muted-foreground">Loading safety metrics...</div>
  }
  
  const { safetyMetrics, pressureMetrics } = coherence
  
  const getSafetyColor = (score: number) => {
    if (score >= 0.8) return 'text-green-500'
    if (score >= 0.6) return 'text-yellow-500'
    return 'text-red-500'
  }
  
  const getRiskLevel = (risk: number) => {
    if (risk < 0.3) return { label: 'Low', color: 'text-green-500' }
    if (risk < 0.6) return { label: 'Medium', color: 'text-yellow-500' }
    return { label: 'High', color: 'text-red-500' }
  }
  
  const howlroundRisk = getRiskLevel(safetyMetrics.howlroundRisk)
  
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Safety Score</span>
        </div>
        <span className={`font-bold ${getSafetyColor(safetyMetrics.safetyScore)}`}>
          {(safetyMetrics.safetyScore * 100).toFixed(0)}%
        </span>
      </div>
      <Progress value={safetyMetrics.safetyScore * 100} />
      
      <div className="space-y-3 pt-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">Howlround Risk</span>
          </div>
          <Badge variant="outline" className={howlroundRisk.color}>
            {howlroundRisk.label}
          </Badge>
        </div>
        
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">Pressure Score</span>
          </div>
          <span className="text-sm font-medium">
            {pressureMetrics.pressureScore.toFixed(2)}
          </span>
        </div>
        
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">Adaptation Rate</span>
          </div>
          <span className="text-sm font-medium">
            {(pressureMetrics.adaptationRate * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      
      {safetyMetrics.interventionNeeded && (
        <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md">
          <div className="flex gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-500 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-medium text-yellow-800 dark:text-yellow-200">
                Intervention Recommended
              </p>
              <p className="text-yellow-700 dark:text-yellow-300 mt-1">
                Your coherence levels suggest taking a break or engaging in grounding activities.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}