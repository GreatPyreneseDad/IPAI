import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { User } from '@/types'
import { authService, LoginRequest, RegisterRequest } from '@/services/auth.service'
import { clearTokens } from '@/services/api'
import { useNavigate } from 'react-router-dom'
import { useToast } from '@/components/ui/use-toast'

interface AuthContextType {
  user: User | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (credentials: LoginRequest) => Promise<void>
  register: (data: RegisterRequest) => Promise<void>
  logout: () => Promise<void>
  updateUser: (user: User) => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

interface AuthProviderProps {
  children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const navigate = useNavigate()
  const { toast } = useToast()
  
  const isAuthenticated = !!user
  
  // Check if user is logged in on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('accessToken')
      if (token) {
        try {
          const currentUser = await authService.getCurrentUser()
          setUser(currentUser)
        } catch (error) {
          clearTokens()
        }
      }
      setIsLoading(false)
    }
    
    checkAuth()
  }, [])
  
  const login = async (credentials: LoginRequest) => {
    try {
      const { user, tokens } = await authService.login(credentials)
      setUser(user)
      navigate('/dashboard')
      toast({
        title: 'Welcome back!',
        description: `Logged in as ${user.username}`,
      })
    } catch (error: any) {
      toast({
        title: 'Login failed',
        description: error.response?.data?.detail || 'Invalid credentials',
        variant: 'destructive',
      })
      throw error
    }
  }
  
  const register = async (data: RegisterRequest) => {
    try {
      const newUser = await authService.register(data)
      toast({
        title: 'Registration successful!',
        description: 'Please check your email to verify your account.',
      })
      navigate('/login')
    } catch (error: any) {
      toast({
        title: 'Registration failed',
        description: error.response?.data?.detail || 'Could not create account',
        variant: 'destructive',
      })
      throw error
    }
  }
  
  const logout = async () => {
    try {
      await authService.logout()
    } finally {
      setUser(null)
      navigate('/login')
      toast({
        title: 'Logged out',
        description: 'You have been successfully logged out.',
      })
    }
  }
  
  const updateUser = (updatedUser: User) => {
    setUser(updatedUser)
  }
  
  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated,
        login,
        register,
        logout,
        updateUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}