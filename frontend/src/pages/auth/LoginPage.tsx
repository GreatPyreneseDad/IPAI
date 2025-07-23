import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import { Eye, EyeOff, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useAuth } from '@/contexts/AuthContext'

const loginSchema = z.object({
  username: z.string().min(1, 'Username is required'),
  password: z.string().min(1, 'Password is required'),
})

type LoginFormValues = z.infer<typeof loginSchema>

export const LoginPage = () => {
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const { login } = useAuth()
  
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
  })
  
  const onSubmit = async (data: LoginFormValues) => {
    setIsLoading(true)
    try {
      await login(data)
    } catch (error) {
      // Error is handled in the context
    } finally {
      setIsLoading(false)
    }
  }
  
  return (
    <div className="p-8">
      <div className="space-y-2 text-center mb-8">
        <h2 className="text-2xl font-bold text-white">Welcome back</h2>
        <p className="text-gray-400">Enter your credentials to access your account</p>
      </div>
      
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="username" className="text-gray-300">Username or Email</Label>
          <Input
            id="username"
            type="text"
            placeholder="john@example.com"
            className="bg-gray-700/50 border-gray-600 text-white placeholder:text-gray-500"
            {...register('username')}
          />
          {errors.username && (
            <p className="text-sm text-red-400">{errors.username.message}</p>
          )}
        </div>
        
        <div className="space-y-2">
          <Label htmlFor="password" className="text-gray-300">Password</Label>
          <div className="relative">
            <Input
              id="password"
              type={showPassword ? 'text' : 'password'}
              placeholder="••••••••"
              className="bg-gray-700/50 border-gray-600 text-white placeholder:text-gray-500 pr-10"
              {...register('password')}
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-300"
            >
              {showPassword ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
            </button>
          </div>
          {errors.password && (
            <p className="text-sm text-red-400">{errors.password.message}</p>
          )}
        </div>
        
        <div className="flex items-center justify-between">
          <Link
            to="/forgot-password"
            className="text-sm text-cyan-400 hover:text-cyan-300"
          >
            Forgot password?
          </Link>
        </div>
        
        <Button
          type="submit"
          disabled={isLoading}
          className="w-full bg-cyan-600 hover:bg-cyan-700 text-white"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Signing in...
            </>
          ) : (
            'Sign in'
          )}
        </Button>
      </form>
      
      <div className="mt-6 text-center text-sm">
        <span className="text-gray-400">Don't have an account? </span>
        <Link to="/register" className="text-cyan-400 hover:text-cyan-300">
          Sign up
        </Link>
      </div>
    </div>
  )
}