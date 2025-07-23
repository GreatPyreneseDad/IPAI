import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(date: string | Date): string {
  const d = new Date(date)
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

export function formatDateTime(date: string | Date): string {
  const d = new Date(date)
  return d.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function getCoherenceLevelColor(level: string): string {
  const colors = {
    critical: 'text-red-500 bg-red-50 border-red-200',
    low: 'text-orange-500 bg-orange-50 border-orange-200',
    moderate: 'text-yellow-500 bg-yellow-50 border-yellow-200',
    high: 'text-green-500 bg-green-50 border-green-200',
    optimal: 'text-cyan-500 bg-cyan-50 border-cyan-200',
  }
  return colors[level.toLowerCase()] || colors.moderate
}

export function truncate(str: string, length: number): string {
  if (str.length <= length) return str
  return str.slice(0, length) + '...'
}