import { NextRequest, NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

export async function GET(request: NextRequest) {
  try {
    // Always use the latest cache file - point to the correct signals directory
    const signalsDir = path.join(process.cwd(), '..', 'data', 'signals')
    const files = await fs.readdir(signalsDir)
    const enhancedFiles = files.filter(f => f.startsWith('enhanced_cache_') && f.endsWith('.json'))
    if (enhancedFiles.length === 0) {
      return NextResponse.json({ error: 'No signals cache found' }, { status: 404 })
    }
    // Sort by timestamp in filename (descending)
    enhancedFiles.sort((a, b) => b.localeCompare(a))
    const latestFile = enhancedFiles[0]
    const filePath = path.join(signalsDir, latestFile)
    const fileContent = await fs.readFile(filePath, 'utf-8')
    const parsed = JSON.parse(fileContent)
    const signals = parsed.response?.data?.signals || []
    return NextResponse.json({ signals })
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
} 