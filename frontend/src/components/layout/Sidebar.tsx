import { Link, useLocation } from 'react-router-dom'
import { Brain, LayoutDashboard, MessageSquare, ClipboardList, BarChart3, User, Settings, ChevronLeft } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

interface SidebarProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Chat', href: '/chat', icon: MessageSquare },
  { name: 'Assessment', href: '/assessment', icon: ClipboardList },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Profile', href: '/profile', icon: User },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export const Sidebar: React.FC<SidebarProps> = ({ open, onOpenChange }) => {
  const location = useLocation()
  
  return (
    <div className={cn(
      "fixed inset-y-0 left-0 z-50 flex flex-col bg-card border-r transition-all duration-300",
      open ? "w-64" : "w-16"
    )}>
      <div className="flex h-16 items-center justify-between px-4 border-b">
        <Link to="/dashboard" className="flex items-center gap-2">
          <Brain className="h-8 w-8 text-primary" />
          {open && <span className="font-bold text-lg">IPAI</span>}
        </Link>
        
        <Button
          variant="ghost"
          size="icon"
          onClick={() => onOpenChange(!open)}
          className={cn("transition-transform", !open && "rotate-180")}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      </div>
      
      <nav className="flex-1 space-y-1 p-2">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href
          return (
            <Link
              key={item.name}
              to={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive 
                  ? "bg-primary text-primary-foreground" 
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                !open && "justify-center"
              )}
            >
              <item.icon className="h-5 w-5" />
              {open && <span>{item.name}</span>}
            </Link>
          )
        })}
      </nav>
      
      {open && (
        <div className="border-t p-4">
          <div className="rounded-lg bg-muted p-3">
            <p className="text-sm font-medium">Coherence Score</p>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-bold">0.85</span>
              <span className="text-xs text-muted-foreground">Moderate</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}