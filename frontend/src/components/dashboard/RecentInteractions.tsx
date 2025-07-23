import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'
import { UserInteraction } from '@/types'
import { formatDateTime } from '@/lib/utils'
import { MessageSquare, ClipboardList, BarChart3, Brain } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

const getInteractionIcon = (type: string) => {
  switch (type) {
    case 'chat':
      return MessageSquare
    case 'assessment':
      return ClipboardList
    case 'analysis':
      return BarChart3
    default:
      return Brain
  }
}

export const RecentInteractions = () => {
  const { data: interactions, isLoading } = useQuery({
    queryKey: ['recent-interactions'],
    queryFn: async () => {
      const response = await api.get<UserInteraction[]>('/users/interactions', {
        params: { limit: 5 },
      })
      return response.data
    },
  })
  
  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="flex items-start gap-3">
            <Skeleton className="h-10 w-10 rounded-full" />
            <div className="flex-1 space-y-2">
              <Skeleton className="h-4 w-[200px]" />
              <Skeleton className="h-3 w-[300px]" />
            </div>
          </div>
        ))}
      </div>
    )
  }
  
  if (!interactions || interactions.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No interactions yet. Start a chat to begin!
      </div>
    )
  }
  
  return (
    <div className="space-y-4">
      {interactions.map((interaction) => {
        const Icon = getInteractionIcon(interaction.interactionType)
        const coherenceChange = interaction.coherenceDelta
        
        return (
          <div key={interaction.id} className="flex items-start gap-3">
            <div className="h-10 w-10 rounded-full bg-muted flex items-center justify-center">
              <Icon className="h-5 w-5 text-muted-foreground" />
            </div>
            
            <div className="flex-1 space-y-1">
              <div className="flex items-center gap-2">
                <p className="text-sm font-medium capitalize">
                  {interaction.interactionType}
                </p>
                <Badge 
                  variant={coherenceChange >= 0 ? 'default' : 'destructive'}
                  className="text-xs"
                >
                  {coherenceChange >= 0 ? '+' : ''}{coherenceChange.toFixed(3)}
                </Badge>
              </div>
              
              <p className="text-sm text-muted-foreground line-clamp-2">
                {interaction.inputText}
              </p>
              
              <p className="text-xs text-muted-foreground">
                {formatDateTime(interaction.createdAt)}
              </p>
            </div>
          </div>
        )
      })}
    </div>
  )
}