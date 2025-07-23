import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Brain, TrendingUp, Shield, Coins, MessageSquare, BarChart3 } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'
import { useCoherence } from '@/contexts/CoherenceContext'
import { getCoherenceLevelColor } from '@/lib/utils'
import { CoherenceChart } from '@/components/dashboard/CoherenceChart'
import { RecentInteractions } from '@/components/dashboard/RecentInteractions'
import { SafetyMetrics } from '@/components/dashboard/SafetyMetrics'
import { useNavigate } from 'react-router-dom'

export const DashboardPage = () => {
  const { user } = useAuth()
  const { currentCoherence, coherenceHistory, isLoading } = useCoherence()
  const navigate = useNavigate()
  
  const coherencePercentage = currentCoherence 
    ? Math.round(currentCoherence.coherenceScore * 100)
    : 0
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground">
          Monitor your coherence and interact with your IPAI
        </p>
      </div>
      
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Coherence Score</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currentCoherence?.coherenceScore.toFixed(2) || '0.00'}</div>
            <Progress value={coherencePercentage} className="mt-2" />
            <p className={`text-xs mt-2 ${currentCoherence ? getCoherenceLevelColor(currentCoherence.level) : ''}`}>
              {currentCoherence?.level || 'Loading...'}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">SAGE Balance</CardTitle>
            <Coins className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{user?.sageBalance.toFixed(2) || '0.00'}</div>
            <p className="text-xs text-muted-foreground mt-1">
              Earned from verified inferences
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Safety Score</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {currentCoherence?.safetyMetrics.safetyScore.toFixed(2) || '1.00'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {currentCoherence?.safetyMetrics.interventionNeeded 
                ? 'Intervention recommended' 
                : 'System stable'}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Growth Rate</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">+12.5%</div>
            <p className="text-xs text-muted-foreground mt-1">
              Last 30 days
            </p>
          </CardContent>
        </Card>
      </div>
      
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Coherence Trend</CardTitle>
            <CardDescription>Your coherence score over time</CardDescription>
          </CardHeader>
          <CardContent>
            <CoherenceChart data={coherenceHistory} />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Safety Metrics</CardTitle>
            <CardDescription>Real-time safety monitoring</CardDescription>
          </CardHeader>
          <CardContent>
            <SafetyMetrics coherence={currentCoherence} />
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>Common tasks and interactions</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          <Button 
            variant="outline" 
            className="justify-start"
            onClick={() => navigate('/chat')}
          >
            <MessageSquare className="mr-2 h-4 w-4" />
            Start Chat Session
          </Button>
          <Button 
            variant="outline" 
            className="justify-start"
            onClick={() => navigate('/assessment')}
          >
            <BarChart3 className="mr-2 h-4 w-4" />
            Take Assessment
          </Button>
          <Button 
            variant="outline" 
            className="justify-start"
            onClick={() => navigate('/analytics')}
          >
            <TrendingUp className="mr-2 h-4 w-4" />
            View Analytics
          </Button>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Recent Interactions</CardTitle>
          <CardDescription>Your latest IPAI interactions</CardDescription>
        </CardHeader>
        <CardContent>
          <RecentInteractions />
        </CardContent>
      </Card>
    </div>
  )
}