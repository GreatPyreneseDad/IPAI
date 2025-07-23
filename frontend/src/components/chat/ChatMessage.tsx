import { ChatMessage as ChatMessageType } from '@/types'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { formatDateTime } from '@/lib/utils'
import { User, Bot } from 'lucide-react'

interface ChatMessageProps {
  message: ChatMessageType
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user'
  
  return (
    <div className={cn('flex gap-3', isUser && 'flex-row-reverse')}>
      <div className={cn(
        'flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-full',
        isUser ? 'bg-primary text-primary-foreground' : 'bg-muted'
      )}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      
      <div className={cn('flex flex-col gap-1', isUser && 'items-end')}>
        <div className={cn(
          'rounded-lg px-3 py-2 text-sm',
          isUser 
            ? 'bg-primary text-primary-foreground' 
            : 'bg-muted'
        )}>
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
        
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>{formatDateTime(message.timestamp)}</span>
          {message.coherenceImpact !== undefined && (
            <Badge 
              variant={message.coherenceImpact >= 0 ? 'default' : 'destructive'}
              className="text-xs h-5"
            >
              {message.coherenceImpact >= 0 ? '+' : ''}{message.coherenceImpact.toFixed(3)}
            </Badge>
          )}
        </div>
      </div>
    </div>
  )
}