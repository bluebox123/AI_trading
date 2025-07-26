'use client'

import { useState } from 'react'
import { AppLayout } from '@/components/Sidebar'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useAuth } from '@/components/providers'
import { 
  User, 
  Settings, 
  Shield, 
  CreditCard,
  Camera,
  Moon,
  Sun,
  Smartphone,
  Mail,
  Bell
} from 'lucide-react'

// Mock user profile data
const mockProfile = {
  fullName: 'John Doe',
  email: 'john.doe@example.com',
  phone: '+91 98765 43210',
  plan: 'pro',
  riskProfile: 'moderate',
  joinedDate: '2024-01-15',
  avatarUrl: '',
  preferences: {
    theme: 'system',
    notifications: {
      email: true,
      sms: true,
      push: true
    }
  }
}

export default function ProfilePage() {
  const { user } = useAuth()
  const [profile, setProfile] = useState(mockProfile)
  const [isEditing, setIsEditing] = useState(false)

  const handleProfileUpdate = (field: string, value: any) => {
    setProfile(prev => ({ ...prev, [field]: value }))
  }

  const handleNotificationUpdate = (type: string, value: boolean) => {
    setProfile(prev => ({
      ...prev,
      preferences: {
        ...prev.preferences,
        notifications: {
          ...prev.preferences.notifications,
          [type]: value
        }
      }
    }))
  }

  return (
    <AppLayout>
      <div className="p-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Profile & Settings</h1>
          <p className="text-muted-foreground">
            Manage your account settings, preferences, and subscription
          </p>
        </div>

        <Tabs defaultValue="profile" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="profile">Profile</TabsTrigger>
            <TabsTrigger value="preferences">Preferences</TabsTrigger>
            <TabsTrigger value="security">Security</TabsTrigger>
            <TabsTrigger value="billing">Billing</TabsTrigger>
          </TabsList>

          {/* Profile Tab */}
          <TabsContent value="profile">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Profile Picture */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <User className="w-5 h-5 mr-2" />
                    Profile Picture
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-center">
                  <div className="relative inline-block mb-4">
                    <Avatar className="w-24 h-24">
                      <AvatarImage src={profile.avatarUrl} />
                      <AvatarFallback className="text-2xl">
                        {profile.fullName.charAt(0)}
                      </AvatarFallback>
                    </Avatar>
                    <Button
                      size="sm"
                      className="absolute -bottom-1 -right-1 rounded-full w-8 h-8 p-0"
                    >
                      <Camera className="w-4 h-4" />
                    </Button>
                  </div>
                  <Button variant="outline" className="w-full">
                    Upload New Picture
                  </Button>
                </CardContent>
              </Card>

              {/* Personal Information */}
              <Card className="lg:col-span-2">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>Personal Information</CardTitle>
                      <CardDescription>Update your personal details</CardDescription>
                    </div>
                    <Button
                      variant="outline"
                      onClick={() => setIsEditing(!isEditing)}
                    >
                      {isEditing ? 'Cancel' : 'Edit'}
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="fullName">Full Name</Label>
                      <Input
                        id="fullName"
                        value={profile.fullName}
                        onChange={(e) => handleProfileUpdate('fullName', e.target.value)}
                        disabled={!isEditing}
                      />
                    </div>
                    <div>
                      <Label htmlFor="email">Email</Label>
                      <Input
                        id="email"
                        type="email"
                        value={profile.email}
                        disabled
                        className="bg-muted"
                      />
                    </div>
                    <div>
                      <Label htmlFor="phone">Phone Number</Label>
                      <Input
                        id="phone"
                        value={profile.phone}
                        onChange={(e) => handleProfileUpdate('phone', e.target.value)}
                        disabled={!isEditing}
                      />
                    </div>
                    <div>
                      <Label htmlFor="riskProfile">Risk Profile</Label>
                      <Select
                        value={profile.riskProfile}
                        onValueChange={(value) => handleProfileUpdate('riskProfile', value)}
                        disabled={!isEditing}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="conservative">Conservative</SelectItem>
                          <SelectItem value="moderate">Moderate</SelectItem>
                          <SelectItem value="aggressive">Aggressive</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  {isEditing && (
                    <div className="flex space-x-2 pt-4">
                      <Button className="flex-1">Save Changes</Button>
                      <Button variant="outline" onClick={() => setIsEditing(false)}>
                        Cancel
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Preferences Tab */}
          <TabsContent value="preferences">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Theme Settings */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Settings className="w-5 h-5 mr-2" />
                    Appearance
                  </CardTitle>
                  <CardDescription>Customize your app appearance</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <Label>Theme</Label>
                      <Select value={profile.preferences.theme}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="light">
                            <div className="flex items-center">
                              <Sun className="w-4 h-4 mr-2" />
                              Light
                            </div>
                          </SelectItem>
                          <SelectItem value="dark">
                            <div className="flex items-center">
                              <Moon className="w-4 h-4 mr-2" />
                              Dark
                            </div>
                          </SelectItem>
                          <SelectItem value="system">System</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Notification Settings */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Bell className="w-5 h-5 mr-2" />
                    Notifications
                  </CardTitle>
                  <CardDescription>Choose how you want to be notified</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Mail className="w-4 h-4" />
                        <span>Email Notifications</span>
                      </div>
                      <Button
                        variant={profile.preferences.notifications.email ? "default" : "outline"}
                        size="sm"
                        onClick={() => handleNotificationUpdate('email', !profile.preferences.notifications.email)}
                      >
                        {profile.preferences.notifications.email ? 'On' : 'Off'}
                      </Button>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Smartphone className="w-4 h-4" />
                        <span>SMS Notifications</span>
                      </div>
                      <Button
                        variant={profile.preferences.notifications.sms ? "default" : "outline"}
                        size="sm"
                        onClick={() => handleNotificationUpdate('sms', !profile.preferences.notifications.sms)}
                      >
                        {profile.preferences.notifications.sms ? 'On' : 'Off'}
                      </Button>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Bell className="w-4 h-4" />
                        <span>Push Notifications</span>
                      </div>
                      <Button
                        variant={profile.preferences.notifications.push ? "default" : "outline"}
                        size="sm"
                        onClick={() => handleNotificationUpdate('push', !profile.preferences.notifications.push)}
                      >
                        {profile.preferences.notifications.push ? 'On' : 'Off'}
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Security Tab */}
          <TabsContent value="security">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Shield className="w-5 h-5 mr-2" />
                  Security Settings
                </CardTitle>
                <CardDescription>Manage your account security</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-semibold">Password</h4>
                    <p className="text-sm text-muted-foreground">Last changed 3 months ago</p>
                  </div>
                  <Button variant="outline">Change Password</Button>
                </div>
                
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-semibold">Two-Factor Authentication</h4>
                    <p className="text-sm text-muted-foreground">SMS verification enabled</p>
                  </div>
                  <Badge variant="secondary" className="text-green-600 bg-green-100">
                    Enabled
                  </Badge>
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-semibold">Active Sessions</h4>
                    <p className="text-sm text-muted-foreground">Manage your active login sessions</p>
                  </div>
                  <Button variant="outline">Manage Sessions</Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Billing Tab */}
          <TabsContent value="billing">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Current Plan */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <CreditCard className="w-5 h-5 mr-2" />
                    Current Plan
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center p-6">
                    <Badge className="mb-4 text-lg px-4 py-2">
                      Pro Plan
                    </Badge>
                    <div className="text-3xl font-bold mb-2">₹2,999</div>
                    <div className="text-muted-foreground mb-4">per month</div>
                    <Button className="w-full mb-2">Upgrade to Premium</Button>
                    <Button variant="outline" className="w-full">
                      Manage Subscription
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Billing Information */}
              <Card>
                <CardHeader>
                  <CardTitle>Billing Information</CardTitle>
                  <CardDescription>Your billing details and payment method</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-semibold">Payment Method</h4>
                        <p className="text-sm text-muted-foreground">•••• •••• •••• 1234</p>
                      </div>
                      <Button variant="outline" size="sm">Change</Button>
                    </div>
                  </div>
                  
                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-semibold">Next Billing Date</h4>
                        <p className="text-sm text-muted-foreground">February 15, 2025</p>
                      </div>
                    </div>
                  </div>

                  <Button variant="outline" className="w-full">
                    View Billing History
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>

        {/* TODO Section */}
        <div className="mt-6 p-4 bg-muted/50 rounded-lg border-2 border-dashed">
          <h3 className="font-semibold mb-2">TODO: Profile Page Implementation</h3>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• Connect to Supabase profile update functions</li>
            <li>• Implement avatar upload to Supabase Storage</li>
            <li>• Add theme switching functionality</li>
            <li>• Integrate with Stripe for billing management</li>
            <li>• Add password reset functionality</li>
            <li>• Implement notification preferences API</li>
            <li>• Add account deletion option</li>
            <li>• Connect to actual user session data</li>
          </ul>
        </div>
      </div>
    </AppLayout>
  )
} 