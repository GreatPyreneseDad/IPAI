import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import { ArrowLeft, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { authService } from '@/services/auth.service'
import { useToast } from '@/components/ui/use-toast'

const forgotPasswordSchema = z.object({
  email: z.string().email('Invalid email address'),
})

type ForgotPasswordFormValues = z.infer<typeof forgotPasswordSchema>

export const ForgotPasswordPage = () => {
  const [isLoading, setIsLoading] = useState(false)
  const [isSubmitted, setIsSubmitted] = useState(false)
  const { toast } = useToast()
  
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<ForgotPasswordFormValues>({
    resolver: zodResolver(forgotPasswordSchema),
  })
  
  const onSubmit = async (data: ForgotPasswordFormValues) => {
    setIsLoading(true)
    try {
      await authService.resetPassword(data)
      setIsSubmitted(true)
      toast({
        title: 'Reset email sent',
        description: 'Check your email for password reset instructions.',
      })
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.response?.data?.detail || 'Failed to send reset email',
        variant: 'destructive',
      })
    } finally {
      setIsLoading(false)
    }
  }
  
  if (isSubmitted) {
    return (
      <div className="p-8">
        <div className="space-y-4 text-center">
          <div className="mx-auto w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          
          <h2 className="text-2xl font-bold text-white">Check your email</h2>
          <p className="text-gray-400 max-w-sm mx-auto">
            We've sent password reset instructions to your email address. 
            Please check your inbox and follow the link to reset your password.
          </p>
          
          <div className="pt-4">
            <Link
              to="/login"
              className="inline-flex items-center text-cyan-400 hover:text-cyan-300"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to login
            </Link>
          </div>
        </div>
      </div>
    )
  }
  
  return (
    <div className="p-8">
      <div className="space-y-2 text-center mb-8">
        <h2 className="text-2xl font-bold text-white">Reset your password</h2>
        <p className="text-gray-400">Enter your email to receive reset instructions</p>
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
        
        <Button
          type="submit"
          disabled={isLoading}
          className="w-full bg-cyan-600 hover:bg-cyan-700 text-white"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Sending...
            </>
          ) : (
            'Send reset email'
          )}
        </Button>
      </form>
      
      <div className="mt-6 text-center">
        <Link
          to="/login"
          className="inline-flex items-center text-sm text-cyan-400 hover:text-cyan-300"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to login
        </Link>
      </div>
    </div>
  )
}