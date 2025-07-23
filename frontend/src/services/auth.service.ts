import { api, setTokens, clearTokens } from './api'
import { User, AuthTokens } from '@/types'

export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  email: string
  username: string
  password: string
  fullName?: string
}

export interface ResetPasswordRequest {
  email: string
}

export interface ConfirmResetRequest {
  token: string
  newPassword: string
}

export interface ChangePasswordRequest {
  currentPassword: string
  newPassword: string
}

class AuthService {
  async login(credentials: LoginRequest): Promise<{ user: User; tokens: AuthTokens }> {
    const formData = new URLSearchParams()
    formData.append('username', credentials.username)
    formData.append('password', credentials.password)
    
    const response = await api.post('/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    })
    
    const tokens = response.data
    setTokens(tokens)
    
    // Get user profile
    const userResponse = await api.get('/users/me')
    
    return {
      user: userResponse.data,
      tokens,
    }
  }
  
  async register(data: RegisterRequest): Promise<User> {
    const response = await api.post('/auth/register', data)
    return response.data
  }
  
  async logout(): Promise<void> {
    try {
      await api.post('/auth/logout')
    } finally {
      clearTokens()
    }
  }
  
  async getCurrentUser(): Promise<User> {
    const response = await api.get('/users/me')
    return response.data
  }
  
  async updateProfile(data: Partial<User>): Promise<User> {
    const response = await api.put('/users/me', data)
    return response.data
  }
  
  async resetPassword(data: ResetPasswordRequest): Promise<void> {
    await api.post('/auth/reset-password', data)
  }
  
  async confirmReset(data: ConfirmResetRequest): Promise<void> {
    await api.post('/auth/confirm-reset', data)
  }
  
  async changePassword(data: ChangePasswordRequest): Promise<void> {
    await api.post('/auth/change-password', data)
  }
  
  async verifyEmail(token: string): Promise<void> {
    await api.post('/auth/verify-email', { token })
  }
  
  async resendVerification(): Promise<void> {
    await api.post('/auth/resend-verification')
  }
}

export const authService = new AuthService()