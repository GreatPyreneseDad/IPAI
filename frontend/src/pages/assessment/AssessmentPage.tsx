import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { ClipboardList, ArrowRight, CheckCircle } from 'lucide-react'

export const AssessmentPage = () => {
  const [isStarted, setIsStarted] = useState(false)
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const totalQuestions = 20
  
  const progress = (currentQuestion / totalQuestions) * 100
  
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Coherence Assessment</h1>
        <p className="text-muted-foreground">
          Evaluate your psychological coherence across multiple dimensions
        </p>
      </div>
      
      {!isStarted ? (
        <Card>
          <CardHeader>
            <CardTitle>Initial Assessment</CardTitle>
            <CardDescription>
              This assessment will help calibrate your IPAI to your unique psychological profile
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <h3 className="font-semibold">What to expect:</h3>
              <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                <li>20 questions about your thoughts, feelings, and behaviors</li>
                <li>Approximately 10-15 minutes to complete</li>
                <li>No right or wrong answers - be authentic</li>
                <li>Results will calibrate your coherence parameters</li>
              </ul>
            </div>
            
            <div className="grid grid-cols-2 gap-4 pt-4">
              <Card className="p-4">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                    <ClipboardList className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="font-semibold">Comprehensive</p>
                    <p className="text-sm text-muted-foreground">Multi-dimensional analysis</p>
                  </div>
                </div>
              </Card>
              
              <Card className="p-4">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-full bg-green-500/10 flex items-center justify-center">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                  </div>
                  <div>
                    <p className="font-semibold">Personalized</p>
                    <p className="text-sm text-muted-foreground">Tailored to your profile</p>
                  </div>
                </div>
              </Card>
            </div>
            
            <Button 
              onClick={() => setIsStarted(true)} 
              className="w-full"
              size="lg"
            >
              Start Assessment
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <CardTitle>Question {currentQuestion + 1} of {totalQuestions}</CardTitle>
                <span className="text-sm text-muted-foreground">{Math.round(progress)}% complete</span>
              </div>
              <Progress value={progress} />
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-center py-12 text-muted-foreground">
              Assessment questions would appear here
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}