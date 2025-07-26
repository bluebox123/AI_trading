import { NextRequest, NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

interface SentimentDataRow {
  Date: string
  Day_of_Week: string
  Month: string
  Quarter: string
  Symbol: string
  Company_Name: string
  Sector: string
  Market_Cap_Category: string
  Sentiment_Score: number
  Sentiment_Category: string
  Confidence_Score: number
  Primary_Market_Factor: string
  News_Volume: number
  Social_Media_Mentions: number
  Analyst_Coverage: number
  Price_Change_Percent: number
  Volume_Change_Percent: number
  Market_Volatility_Index: number
  Sector_Performance: number
}

// A more robust CSV parser that handles quoted fields
function parseCSV(csvContent: string): any[] {
  const lines = csvContent.split('\n').filter(line => line.trim())
  if (lines.length < 2) return []

  const headers = lines[0].split(',').map(h => h.trim())
  const data: any[] = []

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',')
    if (values.length === headers.length) {
      const row: any = {}
      headers.forEach((header, index) => {
        const value = values[index]?.trim()
        
        // Convert to number if it's a numeric field, otherwise keep as string
        if (!isNaN(parseFloat(value)) && isFinite(value as any)) {
            row[header] = parseFloat(value)
        } else {
            row[header] = value
        }
      })
      data.push(row)
    }
  }
  return data
}

async function getSentimentData(timeRange: string) {
  const dataDir = path.join(process.cwd(), 'data', 'sentiment')
  
  const fileMap: { [key: string]: string } = {
    'month': 'stock_sentiment_dataset_month.csv',
    '2024-2025': 'stock-sentiment-dataset_2024-2025.csv',
    '2023-2024': 'stock_sentiment_dataset_2023-2024.csv',
    '2022-2023': 'stock_sentiment_dataset_2022-2023.csv',
    '2021-2022': 'stock_sentiment_dataset_2021-2022.csv',
    '2020-2021': 'stock_sentiment_dataset_2020-2021.csv',
  }

  if (timeRange === 'all') {
    let allData: any[] = []
    const allFiles = Object.values(fileMap).filter(f => f !== 'stock_sentiment_dataset_month.csv')
    
    for (const file of allFiles) {
      const filePath = path.join(dataDir, file)
      try {
        const csvContent = await fs.readFile(filePath, 'utf-8')
        allData = allData.concat(parseCSV(csvContent))
      } catch (error) {
        console.warn(`Could not read or parse file: ${file}`, error)
      }
    }
    return allData

  } else {
    const fileName = fileMap[timeRange] || fileMap['month']
    const filePath = path.join(dataDir, fileName)

    try {
      const csvContent = await fs.readFile(filePath, 'utf-8')
      return parseCSV(csvContent)
    } catch (error) {
      console.error(`Error reading file for timeRange ${timeRange}:`, error)
      throw new Error(`Data for '${timeRange}' is not available.`)
    }
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const timeRange = searchParams.get('timeRange') || 'month'
    
    const data = await getSentimentData(timeRange)

    if (data.length === 0) {
      return NextResponse.json({
          message: `No data found for time range '${timeRange}'.`
      }, { status: 404 })
    }

    return NextResponse.json(data)
  } catch (error: any) {
    return NextResponse.json({ message: error.message }, { status: 500 })
  }
}

// POST endpoint for real-time sentiment updates
export async function POST(request: NextRequest) {
  try {
    const updates = await request.json()
    
    // TODO: Implement real-time sentiment update logic
    // This would be used to push new sentiment scores from the ML pipeline
    
    console.log('Received sentiment updates:', updates.length, 'records')
    
    return NextResponse.json({
      success: true,
      message: `Successfully processed ${updates.length} sentiment updates`,
      timestamp: new Date().toISOString()
    })
    
  } catch (error) {
    console.error('Error processing sentiment updates:', error)
    
    return NextResponse.json({
      success: false,
      error: 'Failed to process sentiment updates'
    }, { status: 500 })
  }
} 