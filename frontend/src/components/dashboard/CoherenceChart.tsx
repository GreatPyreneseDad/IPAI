import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js'
import { CoherenceProfile } from '@/types'
import { formatDate } from '@/lib/utils'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

interface CoherenceChartProps {
  data: CoherenceProfile[]
}

export const CoherenceChart: React.FC<CoherenceChartProps> = ({ data }) => {
  const chartData = {
    labels: data.map(d => formatDate(d.calculatedAt)),
    datasets: [
      {
        label: 'Coherence Score',
        data: data.map(d => d.coherenceScore),
        borderColor: 'rgb(6, 182, 212)',
        backgroundColor: 'rgba(6, 182, 212, 0.5)',
        tension: 0.3,
      },
      {
        label: 'Soul Echo',
        data: data.map(d => d.soulEcho),
        borderColor: 'rgb(168, 85, 247)',
        backgroundColor: 'rgba(168, 85, 247, 0.5)',
        tension: 0.3,
      },
    ],
  }
  
  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
      },
    },
  }
  
  return (
    <div className="h-[300px]">
      <Line data={chartData} options={options} />
    </div>
  )
}