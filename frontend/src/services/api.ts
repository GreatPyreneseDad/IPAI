import axios, { AxiosError, AxiosRequestConfig } from 'axios'
import { AuthTokens } from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

// Create axios instance
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Token management
let accessToken: string | null = localStorage.getItem('accessToken')
let refreshToken: string | null = localStorage.getItem('refreshToken')

export const setTokens = (tokens: AuthTokens) => {
  accessToken = tokens.accessToken
  refreshToken = tokens.refreshToken
  localStorage.setItem('accessToken', tokens.accessToken)
  localStorage.setItem('refreshToken', tokens.refreshToken)
}

export const clearTokens = () => {
  accessToken = null
  refreshToken = null
  localStorage.removeItem('accessToken')
  localStorage.removeItem('refreshToken')
}

// Request interceptor
api.interceptors.request.use(
  (config) => {
    if (accessToken) {
      config.headers.Authorization = `Bearer ${accessToken}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean }
    
    if (error.response?.status === 401 && !originalRequest._retry && refreshToken) {
      originalRequest._retry = true
      
      try {
        const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
          refreshToken,
        })
        
        const tokens = response.data
        setTokens(tokens)
        
        originalRequest.headers = {
          ...originalRequest.headers,
          Authorization: `Bearer ${tokens.accessToken}`,
        }
        
        return api(originalRequest)
      } catch (refreshError) {
        clearTokens()
        window.location.href = '/login'
        return Promise.reject(refreshError)
      }
    }
    
    return Promise.reject(error)
  }
)

// Generic error handler
export const handleApiError = (error: any): string => {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      // Server responded with error
      return error.response.data.detail || error.response.data.message || 'An error occurred'
    } else if (error.request) {
      // Request made but no response
      return 'Network error. Please check your connection.'
    }
  }
  return error.message || 'An unexpected error occurred'
}