import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Brain, Shield, Bell, Wallet, Key } from 'lucide-react'

export const SettingsPage = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground">
          Configure your IPAI experience and preferences
        </p>
      </div>
      
      <Tabs defaultValue="general" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="llm">LLM Providers</TabsTrigger>
          <TabsTrigger value="wallet">Wallet</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
        </TabsList>
        
        <TabsContent value="general" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Notifications</CardTitle>
              <CardDescription>Configure how you receive alerts and updates</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="coherence-alerts">Coherence Alerts</Label>
                  <p className="text-sm text-muted-foreground">
                    Get notified when coherence levels require attention
                  </p>
                </div>
                <Switch id="coherence-alerts" defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="achievement-notifications">Achievement Notifications</Label>
                  <p className="text-sm text-muted-foreground">
                    Celebrate milestones and earned badges
                  </p>
                </div>
                <Switch id="achievement-notifications" defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="email-updates">Email Updates</Label>
                  <p className="text-sm text-muted-foreground">
                    Receive weekly coherence reports via email
                  </p>
                </div>
                <Switch id="email-updates" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Privacy</CardTitle>
              <CardDescription>Control your data and privacy settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="anonymous-analytics">Anonymous Analytics</Label>
                  <p className="text-sm text-muted-foreground">
                    Help improve IPAI by sharing anonymous usage data
                  </p>
                </div>
                <Switch id="anonymous-analytics" />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="public-profile">Public Profile</Label>
                  <p className="text-sm text-muted-foreground">
                    Allow others to view your coherence achievements
                  </p>
                </div>
                <Switch id="public-profile" />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="llm" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>LLM Provider Configuration</CardTitle>
              <CardDescription>Connect and manage AI language models</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-center py-8 text-muted-foreground">
                LLM provider settings would be configured here
              </p>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="wallet" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Wallet Connection</CardTitle>
              <CardDescription>Manage your crypto wallet for SAGE tokens</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-center py-8 text-muted-foreground">
                Wallet connection interface would be here
              </p>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="security" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Security Settings</CardTitle>
              <CardDescription>Protect your account and data</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button variant="outline" className="w-full justify-start">
                <Key className="mr-2 h-4 w-4" />
                Change Password
              </Button>
              
              <Button variant="outline" className="w-full justify-start">
                <Shield className="mr-2 h-4 w-4" />
                Enable Two-Factor Authentication
              </Button>
              
              <div className="pt-4 border-t">
                <h4 className="text-sm font-medium mb-2">Active Sessions</h4>
                <p className="text-sm text-muted-foreground">
                  Manage devices with access to your account
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}