/**
 * C++ Source Code Parser
 * 
 * Extracts function/class/struct definitions from C++ source files
 */

import fs from 'fs'
import path from 'path'

export interface SourceCodeLocation {
  file: string
  startLine: number
  endLine: number
  code: string
}

export interface ParseOptions {
  projectRoot?: string
  searchPaths?: string[]
}

/**
 * Find a file in the project
 */
function findFile(filename: string, searchPaths: string[]): string | null {
  for (const searchPath of searchPaths) {
    const fullPath = path.join(searchPath, filename)
    if (fs.existsSync(fullPath)) {
      return fullPath
    }
  }
  return null
}

/**
 * Read file content with line numbers
 */
function readFileLines(filePath: string): string[] {
  const content = fs.readFileSync(filePath, 'utf-8')
  return content.split('\n')
}

/**
 * Find matching brace for a given position
 */
function findMatchingBrace(lines: string[], startLine: number, startCol: number): number {
  let braceCount = 0
  let inString = false
  let inChar = false
  let inComment = false
  let inBlockComment = false
  
  for (let i = startLine; i < lines.length; i++) {
    const line = lines[i]
    const startIdx = (i === startLine) ? startCol : 0
    
    for (let j = startIdx; j < line.length; j++) {
      const char = line[j]
      const nextChar = line[j + 1]
      
      // Handle comments
      if (!inString && !inChar) {
        if (char === '/' && nextChar === '/') {
          inComment = true
          break // Skip rest of line
        }
        if (char === '/' && nextChar === '*') {
          inBlockComment = true
          j++ // Skip next char
          continue
        }
        if (inBlockComment && char === '*' && nextChar === '/') {
          inBlockComment = false
          j++ // Skip next char
          continue
        }
      }
      
      if (inBlockComment || inComment) continue
      
      // Handle strings
      if (char === '"' && (j === 0 || line[j - 1] !== '\\')) {
        inString = !inString
        continue
      }
      if (char === "'" && (j === 0 || line[j - 1] !== '\\')) {
        inChar = !inChar
        continue
      }
      
      if (inString || inChar) continue
      
      // Count braces
      if (char === '{') {
        braceCount++
      } else if (char === '}') {
        braceCount--
        if (braceCount === 0) {
          return i
        }
      }
    }
    
    inComment = false // Reset line comment at end of line
  }
  
  return -1 // Not found
}

/**
 * Find symbol definition in lines
 */
function findSymbolDefinition(
  lines: string[], 
  symbolName: string,
  symbolType?: 'function' | 'class' | 'struct' | 'enum'
): { startLine: number; endLine: number } | null {
  
  // Build regex patterns for different symbol types
  const patterns: RegExp[] = []
  
  if (!symbolType || symbolType === 'function') {
    // Function pattern: return_type function_name(...) or template<...> return_type function_name(...)
    patterns.push(
      new RegExp(`^\\s*(?:template\\s*<[^>]*>\\s*)?(?:\\w+\\s+)+${symbolName}\\s*\\(`),
      new RegExp(`^\\s*(?:inline\\s+)?(?:static\\s+)?(?:constexpr\\s+)?(?:\\w+\\s+)+${symbolName}\\s*\\(`),
      new RegExp(`^\\s*${symbolName}\\s*\\(`) // Constructor
    )
  }
  
  if (!symbolType || symbolType === 'class') {
    patterns.push(
      new RegExp(`^\\s*(?:template\\s*<[^>]*>\\s*)?class\\s+${symbolName}\\s*[:{]`),
      new RegExp(`^\\s*class\\s+${symbolName}\\s*;`) // Forward declaration
    )
  }
  
  if (!symbolType || symbolType === 'struct') {
    patterns.push(
      new RegExp(`^\\s*(?:template\\s*<[^>]*>\\s*)?struct\\s+${symbolName}\\s*[:{]`),
      new RegExp(`^\\s*struct\\s+${symbolName}\\s*;`) // Forward declaration
    )
  }
  
  if (!symbolType || symbolType === 'enum') {
    patterns.push(
      new RegExp(`^\\s*enum\\s+(?:class\\s+)?${symbolName}\\s*[:{]`)
    )
  }
  
  // Search for symbol
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    
    // Skip comments and empty lines
    if (line.trim().startsWith('//') || line.trim() === '') continue
    
    for (const pattern of patterns) {
      if (pattern.test(line)) {
        // Found symbol declaration
        const startLine = i
        
        // Check if it's a forward declaration (ends with ;)
        if (line.trim().endsWith(';')) {
          return { startLine, endLine: i }
        }
        
        // Find opening brace
        let braceCol = line.indexOf('{')
        let braceLine = i
        
        if (braceCol === -1) {
          // Look for opening brace in next lines
          for (let j = i + 1; j < Math.min(i + 20, lines.length); j++) {
            braceCol = lines[j].indexOf('{')
            if (braceCol !== -1) {
              braceLine = j
              break
            }
            // If we hit a semicolon, it's just a declaration
            if (lines[j].includes(';')) {
              return { startLine, endLine: j }
            }
          }
        }
        
        if (braceCol !== -1) {
          // Find matching closing brace
          const endLine = findMatchingBrace(lines, braceLine, braceCol)
          if (endLine !== -1) {
            return { startLine, endLine }
          }
        }
        
        // If no braces found, assume single-line or declaration
        return { startLine, endLine: i }
      }
    }
  }
  
  return null
}

/**
 * Extract source code for a symbol
 */
export function extractSourceCode(
  filePath: string,
  symbolName: string,
  options: ParseOptions = {}
): SourceCodeLocation | null {
  
  const projectRoot = options.projectRoot || process.cwd()
  const searchPaths = options.searchPaths || [
    projectRoot,
    path.join(projectRoot, 'scl'),
    path.join(projectRoot, 'scl/core'),
    path.join(projectRoot, 'scl/kernel'),
    path.join(projectRoot, 'scl/math'),
    path.join(projectRoot, 'scl/threading')
  ]
  
  // Find the file
  const fullPath = findFile(filePath, searchPaths)
  if (!fullPath) {
    throw new Error(`File not found: ${filePath} (searched in: ${searchPaths.join(', ')})`)
  }
  
  // Read file
  const lines = readFileLines(fullPath)
  
  // Find symbol
  const location = findSymbolDefinition(lines, symbolName)
  if (!location) {
    throw new Error(`Symbol '${symbolName}' not found in ${filePath}`)
  }
  
  // Extract code
  const codeLines = lines.slice(location.startLine, location.endLine + 1)
  const code = codeLines.join('\n')
  
  return {
    file: fullPath,
    startLine: location.startLine + 1, // 1-indexed for display
    endLine: location.endLine + 1,
    code
  }
}

/**
 * Parse source_code block parameters
 */
export interface SourceCodeParams {
  file: string
  symbol: string
  lang?: string
  title?: string
  highlight?: string
}

export function parseSourceCodeParams(paramString: string): SourceCodeParams {
  const params: SourceCodeParams = {
    file: '',
    symbol: '',
    lang: 'cpp'
  }
  
  // Parse key=value pairs
  const regex = /(\w+)="([^"]*)"/g
  let match
  
  while ((match = regex.exec(paramString)) !== null) {
    const key = match[1]
    const value = match[2]
    
    if (key === 'file') params.file = value
    else if (key === 'symbol') params.symbol = value
    else if (key === 'lang') params.lang = value
    else if (key === 'title') params.title = value
    else if (key === 'highlight') params.highlight = value
  }
  
  if (!params.file || !params.symbol) {
    throw new Error('source_code block requires both file and symbol parameters')
  }
  
  return params
}

