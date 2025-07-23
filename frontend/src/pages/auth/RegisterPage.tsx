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

const registerSchema = z.object({
  email: z.string().email('Invalid email address'),
  username: z.string()
    .min(3, 'Username must be at least 3 characters')
    .max(20, 'Username must be less than 20 characters')
    .regex(/^[a-zA-Z0-9_]+$/, 'Username can only contain letters, numbers, and underscores'),
  fullName: z.string().optional(),
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
    .regex(/[0-9]/, 'Password must contain at least one number'),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
})

type RegisterFormValues = z.infer<typeof registerSchema>

export const RegisterPage = () => {
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const { register: registerUser } = useAuth()
  
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<RegisterFormValues>({
    resolver: zodResolver(registerSchema),
  })
  
  const onSubmit = async (data: RegisterFormValues) => {
    setIsLoading(true)
    try {
      await registerUser({
        email: data.email,
        username: data.username,
        password: data.password,
        fullName: data.fullName,
      })
    } catch (error) {
      // Error is handled in the context
    } finally {
      setIsLoading(false)
    }
  }
  
  return (
    <div className="p-8">
      <div className="space-y-2 text-center mb-8">
        <h2 className="text-2xl font-bold text-white">Create your account</h2>
        <p className="text-gray-400">Join IPAI to preserve your authentic consciousness</p>
      </div>
      
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="email" className="text-gray-300">Email</Label>
          <Input
            id="email"
            type="email"
            placeholder="john@example.com"
            className="bg-gray-700/50 border-gray-600 text-white placeholder:text-gray-500"
            {...register('email')}
          />
          {errors.email && (
            <p className="text-sm text-red-400">{errors.email.message}</p>
          )}
        </div>
        
        <div className="space-y-2">
          <Label htmlFor="username" className="text-gray-300">Username</Label>
          <Input
            id="username"
            type="text"
            placeholder="johndoe"
            className="bg-gray-700/50 border-gray-600 text-white placeholder:text-gray-500"
            {...register('username')}
          />
          {errors.username && (
            <p className="text-sm text-red-400">{errors.username.message}</p>
          )}
        </div>
        
        <div className="space-y-2">
          <Label htmlFor="fullName" className="text-gray-300">Full Name (optional)</Label>
          <Input
            id="fullName"
            type="text"
            placeholder="John Doe"
            className="bg-gray-700/50 border-gray-600 text-white placeholder:text-gray-500"
            {...register('fullName')}
          />
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
        
        <div className="space-y-2">
          <Label htmlFor="confirmPassword" className="text-gray-300">Confirm Password</Label>
          <div className="relative">
            <Input
              id="confirmPassword"
              type={showConfirmPassword ? 'text' : 'password'}
              placeholder="••••••••"
              className="bg-gray-700/50 border-gray-600 text-white placeholder:text-gray-500 pr-10"
              {...register('confirmPassword')}
            />
            <button
              type="button"
              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-300"
            >
              {showConfirmPassword ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
            </button>
          </div>
          {errors.confirmPassword && (
            <p className="text-sm text-red-400">{errors.confirmPassword.message}</p>
          )}
        </div>
        
        <Button
          type="submit"
          disabled={isLoading}
          className="w-full bg-cyan-600 hover:bg-cyan-700 text-white"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Creating account...
            </>
          ) : (
            'Create account'
          )}
        </Button>
      </form>
      
      <div className="mt-6 text-center text-sm">
        <span className="text-gray-400">Already have an account? </span>
        <Link to="/login" className="text-cyan-400 hover:text-cyan-300">
          Sign in
        </Link>
      </div>
    </div>
  )
}