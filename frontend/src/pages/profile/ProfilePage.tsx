import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Badge } from '@/components/ui/badge'
import { useAuth } from '@/contexts/AuthContext'
import { Save, Camera } from 'lucide-react'
import { useToast } from '@/components/ui/use-toast'

export const ProfilePage = () => {
  const { user, updateUser } = useAuth()
  const { toast } = useToast()
  const [isEditing, setIsEditing] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  
  const [formData, setFormData] = useState({
    fullName: user?.fullName || '',
    bio: user?.bio || '',
    email: user?.email || '',
  })
  
  const handleSave = async () => {
    setIsLoading(true)
    try {
      // API call would go here
      toast({
        title: 'Profile updated',
        description: 'Your profile has been successfully updated.',
      })
      setIsEditing(false)
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to update profile.',
        variant: 'destructive',
      })
    } finally {
      setIsLoading(false)
    }
  }
  
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Profile</h1>
        <p className="text-muted-foreground">
          Manage your account information and preferences
        </p>
      </div>
      
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Personal Information</CardTitle>
              <CardDescription>Update your profile details</CardDescription>
            </div>
            {!isEditing ? (
              <Button onClick={() => setIsEditing(true)}>Edit Profile</Button>
            ) : (
              <div className="flex gap-2">
                <Button variant="outline" onClick={() => setIsEditing(false)}>
                  Cancel
                </Button>
                <Button onClick={handleSave} disabled={isLoading}>
                  <Save className="mr-2 h-4 w-4" />
                  Save Changes
                </Button>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center gap-6">
            <div className="relative">
              <Avatar className="h-24 w-24">
                <AvatarImage src={user?.avatarUrl} />
                <AvatarFallback>
                  {user?.username?.charAt(0).toUpperCase()}
                </AvatarFallback>
              </Avatar>
              {isEditing && (
                <Button
                  size="icon"
                  variant="secondary"
                  className="absolute bottom-0 right-0 h-8 w-8 rounded-full"
                >
                  <Camera className="h-4 w-4" />
                </Button>
              )}
            </div>
            
            <div>
              <h3 className="text-lg font-semibold">{user?.username}</h3>
              <div className="flex items-center gap-2 mt-1">
                <Badge variant="outline">{user?.role}</Badge>
                <Badge variant="outline" className={user?.isVerified ? 'text-green-600' : 'text-yellow-600'}>
                  {user?.isVerified ? 'Verified' : 'Unverified'}
                </Badge>
              </div>
            </div>
          </div>
          
          <div className="grid gap-4">
            <div className="grid gap-2">
              <Label htmlFor="fullName">Full Name</Label>
              {isEditing ? (
                <Input
                  id="fullName"
                  value={formData.fullName}
                  onChange={(e) => setFormData({ ...formData, fullName: e.target.value })}
                />
              ) : (
                <p className="text-sm">{user?.fullName || 'Not set'}</p>
              )}
            </div>
            
            <div className="grid gap-2">
              <Label htmlFor="email">Email</Label>
              <p className="text-sm">{user?.email}</p>
              {!user?.isVerified && (
                <Button variant="link" className="justify-start p-0 h-auto text-sm">
                  Verify email address
                </Button>
              )}
            </div>
            
            <div className="grid gap-2">
              <Label htmlFor="bio">Bio</Label>
              {isEditing ? (
                <Textarea
                  id="bio"
                  value={formData.bio}
                  onChange={(e) => setFormData({ ...formData, bio: e.target.value })}
                  placeholder="Tell us about yourself..."
                  className="min-h-[100px]"
                />
              ) : (
                <p className="text-sm">{user?.bio || 'No bio added yet'}</p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Account Statistics</CardTitle>
          <CardDescription>Your IPAI journey at a glance</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Member Since</p>
              <p className="text-2xl font-bold">
                {user?.createdAt ? new Date(user.createdAt).toLocaleDateString() : 'N/A'}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">SAGE Balance</p>
              <p className="text-2xl font-bold">{user?.sageBalance.toFixed(2) || '0.00'}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Coherence Level</p>
              <p className="text-2xl font-bold capitalize">{user?.coherenceLevel || 'Unknown'}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}