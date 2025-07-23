import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { TrendingUp, TrendingDown, Activity, Brain } from 'lucide-react'
import { CoherenceChart } from '@/components/dashboard/CoherenceChart'
import { useCoherence } from '@/contexts/CoherenceContext'
import { useState } from 'react'

export const AnalyticsPage = () => {
  const { coherenceHistory } = useCoherence()
  const [timeRange, setTimeRange] = useState('30')
  
  // Calculate statistics
  const avgCoherence = coherenceHistory.length > 0
    ? coherenceHistory.reduce((sum, c) => sum + c.coherenceScore, 0) / coherenceHistory.length
    : 0
  
  const trend = coherenceHistory.length > 1
    ? coherenceHistory[coherenceHistory.length - 1].coherenceScore - coherenceHistory[0].coherenceScore
    : 0
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Analytics</h1>
          <p className="text-muted-foreground">
            Deep insights into your coherence patterns
          </p>
        </div>
        
        <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select time range" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="7">Last 7 days</SelectItem>
            <SelectItem value="30">Last 30 days</SelectItem>
            <SelectItem value="90">Last 90 days</SelectItem>
            <SelectItem value="365">Last year</SelectItem>
          </SelectContent>
        </Select>
      </div>
      
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Average Coherence</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{avgCoherence.toFixed(3)}</div>
            <p className="text-xs text-muted-foreground">
              Last {timeRange} days
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Trend</CardTitle>
            {trend >= 0 ? (
              <TrendingUp className="h-4 w-4 text-green-500" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-500" />
            )}
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {trend >= 0 ? '+' : ''}{trend.toFixed(3)}
            </div>
            <p className="text-xs text-muted-foreground">
              Change over period
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Interactions</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">247</div>
            <p className="text-xs text-muted-foreground">
              Last {timeRange} days
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Stability Index</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0.82</div>
            <p className="text-xs text-muted-foreground">
              Higher is better
            </p>
          </CardContent>
        </Card>
      </div>
      
      <Tabs defaultValue="coherence" className="space-y-4">
        <TabsList>
          <TabsTrigger value="coherence">Coherence Trends</TabsTrigger>
          <TabsTrigger value="components">Component Analysis</TabsTrigger>
          <TabsTrigger value="patterns">Behavior Patterns</TabsTrigger>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
        </TabsList>
        
        <TabsContent value="coherence" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Coherence Over Time</CardTitle>
              <CardDescription>
                Track your psychological coherence and soul echo trends
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CoherenceChart data={coherenceHistory} />
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="components" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>GCT Component Breakdown</CardTitle>
              <CardDescription>
                Analysis of your four core coherence components
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-center py-12 text-muted-foreground">
                Component analysis visualization would appear here
              </p>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="patterns" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Behavioral Patterns</CardTitle>
              <CardDescription>
                Identify patterns in your interactions and coherence changes
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-center py-12 text-muted-foreground">
                Pattern analysis would appear here
              </p>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI-Generated Insights</CardTitle>
              <CardDescription>
                Personalized recommendations based on your coherence data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-center py-12 text-muted-foreground">
                AI insights would appear here
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}