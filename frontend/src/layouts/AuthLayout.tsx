import { Outlet, Link } from 'react-router-dom'
import { Brain } from 'lucide-react'

export const AuthLayout = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex flex-col">
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="text-center mb-8">
            <Link to="/" className="inline-flex items-center justify-center gap-3 mb-6">
              <Brain className="h-12 w-12 text-cyan-400" />
              <h1 className="text-4xl font-bold text-white">IPAI</h1>
            </Link>
            <p className="text-gray-400">
              Individually Programmed AI - Preserving Authentic Consciousness
            </p>
          </div>
          
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg shadow-xl">
            <Outlet />
          </div>
          
          <div className="mt-8 text-center text-sm text-gray-500">
            <p>
              By using IPAI, you agree to our{' '}
              <Link to="/terms" className="text-cyan-400 hover:text-cyan-300">
                Terms of Service
              </Link>{' '}
              and{' '}
              <Link to="/privacy" className="text-cyan-400 hover:text-cyan-300">
                Privacy Policy
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}