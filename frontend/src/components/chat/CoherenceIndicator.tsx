import { Card } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Brain, TrendingUp, TrendingDown } from 'lucide-react'
import { CoherenceSnapshot } from '@/services/coherence.service'
import { getCoherenceLevelColor, cn } from '@/lib/utils'

interface CoherenceIndicatorProps {
  coherence: CoherenceSnapshot | null
}

export const CoherenceIndicator: React.FC<CoherenceIndicatorProps> = ({ coherence }) => {
  if (!coherence) {
    return (
      <Card className="p-3 w-64">
        <div className="text-sm text-muted-foreground">Loading coherence...</div>
      </Card>
    )
  }
  
  const percentage = Math.round(coherence.coherenceScore * 100)
  const trend = coherence.pressureMetrics.adaptationRate > 0.5 ? 'up' : 'down'
  
  return (
    <Card className="p-3 w-64">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Live Coherence</span>
          </div>
          {trend === 'up' ? (
            <TrendingUp className="h-4 w-4 text-green-500" />
          ) : (
            <TrendingDown className="h-4 w-4 text-red-500" />
          )}
        </div>
        
        <Progress value={percentage} className="h-2" />
        
        <div className="flex items-center justify-between">
          <span className="text-lg font-bold">{coherence.coherenceScore.toFixed(3)}</span>
          <Badge 
            variant="outline" 
            className={cn('text-xs', getCoherenceLevelColor(coherence.level))}
          >
            {coherence.level}
          </Badge>
        </div>
      </div>
    </Card>
  )
}