import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ChatMessage as ChatMessageType } from '@/types'
import { llmService } from '@/services/llm.service'
import { useCoherence } from '@/contexts/CoherenceContext'
import { useToast } from '@/components/ui/use-toast'
import { ChatMessage } from '@/components/chat/ChatMessage'
import { CoherenceIndicator } from '@/components/chat/CoherenceIndicator'
import { cn } from '@/lib/utils'

export const ChatPage = () => {
  const [messages, setMessages] = useState<ChatMessageType[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { currentCoherence, refreshCoherence } = useCoherence()
  const { toast } = useToast()
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }
  
  useEffect(() => {
    scrollToBottom()
  }, [messages])
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return
    
    const userMessage: ChatMessageType = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString(),
    }
    
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    
    try {
      const response = await llmService.chat({
        message: userMessage.content,
        context: messages.slice(-10), // Last 10 messages for context
      })
      
      const assistantMessage: ChatMessageType = {
        role: 'assistant',
        content: response.message,
        timestamp: new Date().toISOString(),
        coherenceImpact: response.coherenceImpact,
        metadata: response.metadata,
      }
      
      setMessages(prev => [...prev, assistantMessage])
      
      // Refresh coherence after interaction
      await refreshCoherence()
      
      // Show alert if coherence impact is significant
      if (Math.abs(response.coherenceImpact) > 0.05) {
        toast({
          title: response.coherenceImpact > 0 ? 'Coherence Improved!' : 'Coherence Warning',
          description: `Your coherence changed by ${response.coherenceImpact > 0 ? '+' : ''}${response.coherenceImpact.toFixed(3)}`,
          variant: response.coherenceImpact > 0 ? 'default' : 'destructive',
        })
      }
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to send message',
        variant: 'destructive',
      })
    } finally {
      setIsLoading(false)
    }
  }
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as any)
    }
  }
  
  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold">Chat with IPAI</h1>
          <p className="text-muted-foreground">
            Engage in coherence-aware conversations
          </p>
        </div>
        
        <CoherenceIndicator coherence={currentCoherence} />
      </div>
      
      <Card className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <p className="mb-2">Start a conversation with your IPAI</p>
              <p className="text-sm">Your coherence is being tracked in real-time</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="border-t p-4">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
              className="min-h-[60px] resize-none"
              disabled={isLoading}
            />
            <Button 
              type="submit" 
              disabled={!input.trim() || isLoading}
              className="self-end"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </form>
          
          {currentCoherence?.safetyMetrics.interventionNeeded && (
            <div className="mt-3 flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-500">
              <AlertTriangle className="h-4 w-4" />
              <span>High coherence variance detected. Consider taking a break.</span>
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}