import { Outlet } from 'react-router-dom'
import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import { useState } from 'react'

export const MainLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  
  return (
    <div className="min-h-screen bg-background">
      <Sidebar open={sidebarOpen} onOpenChange={setSidebarOpen} />
      
      <div className={`transition-all duration-300 ${sidebarOpen ? 'pl-64' : 'pl-16'}`}>
        <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
        
        <main className="p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}