/**
 * Vite Plugin for SCL Source Code Extraction
 * 
 * This plugin provides a virtual module system for extracting C++ source code
 * from the SCL codebase at build time.
 * 
 * Usage in Vue components:
 *   import { code, file, startLine, endLine } from 'virtual:scl-source?file=scl/core/memory.hpp&symbol=aligned_alloc'
 * 
 * Or use the SourceCode component:
 *   <SourceCode file="scl/core/memory.hpp" symbol="aligned_alloc" />
 */

import type { Plugin, ResolvedConfig } from 'vite'
import fs from 'fs'
import path from 'path'

// =============================================================================
// Types
// =============================================================================

interface SourceCodeResult {
  code: string
  file: string
  startLine: number
  endLine: number
  symbol: string
}

interface PluginOptions {
  /** Project root directory (where scl/ folder is located) */
  projectRoot?: string
  /** Additional search paths for source files */
  searchPaths?: string[]
}

// =============================================================================
// C++ Parser
// =============================================================================

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
 * Find matching brace for a given position
 */
function findMatchingBrace(lines: string[], startLine: number, startCol: number): number {
  let braceCount = 0
  let inString = false
  let inChar = false
  let inBlockComment = false
  
  for (let i = startLine; i < lines.length; i++) {
    const line = lines[i]
    const startIdx = (i === startLine) ? startCol : 0
    
    for (let j = startIdx; j < line.length; j++) {
      const char = line[j]
      const nextChar = line[j + 1]
      const prevChar = j > 0 ? line[j - 1] : ''
      
      // Handle block comments
      if (!inString && !inChar) {
        if (char === '/' && nextChar === '*' && !inBlockComment) {
          inBlockComment = true
          j++
          continue
        }
        if (char === '*' && nextChar === '/' && inBlockComment) {
          inBlockComment = false
          j++
          continue
        }
        // Skip line comments
        if (char === '/' && nextChar === '/') {
          break
        }
      }
      
      if (inBlockComment) continue
      
      // Handle strings
      if (char === '"' && prevChar !== '\\') {
        inString = !inString
        continue
      }
      if (char === "'" && prevChar !== '\\') {
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
  }
  
  return -1
}

/**
 * Find symbol definition in lines
 */
function findSymbolDefinition(
  lines: string[], 
  symbolName: string
): { startLine: number; endLine: number } | null {
  
  // Escape special regex characters in symbol name
  const escapedSymbol = symbolName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  
  // Patterns to match different C++ constructs
  const patterns: RegExp[] = [
    // Template function/class
    new RegExp(`^\\s*template\\s*<[^>]*>\\s*(?:inline\\s+)?(?:static\\s+)?(?:constexpr\\s+)?(?:[\\w:]+\\s+)*${escapedSymbol}\\s*[<(]`),
    // Regular function
    new RegExp(`^\\s*(?:inline\\s+)?(?:static\\s+)?(?:constexpr\\s+)?(?:SCL_\\w+\\s+)?(?:[\\w:*&<>]+\\s+)+${escapedSymbol}\\s*\\(`),
    // Class/struct
    new RegExp(`^\\s*(?:template\\s*<[^>]*>\\s*)?(?:class|struct)\\s+${escapedSymbol}\\s*[:{<]`),
    // Enum
    new RegExp(`^\\s*enum\\s+(?:class\\s+)?${escapedSymbol}\\s*[:{]`),
    // Type alias
    new RegExp(`^\\s*using\\s+${escapedSymbol}\\s*=`),
    // Variable/constant
    new RegExp(`^\\s*(?:inline\\s+)?(?:static\\s+)?(?:constexpr\\s+)?(?:[\\w:*&<>]+\\s+)+${escapedSymbol}\\s*[=;{]`),
    // Macro
    new RegExp(`^\\s*#define\\s+${escapedSymbol}[\\s(]`),
  ]
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    
    // Skip pure comment lines
    const trimmed = line.trim()
    if (trimmed.startsWith('//') || trimmed.startsWith('/*') || trimmed.startsWith('*')) {
      continue
    }
    
    for (const pattern of patterns) {
      if (pattern.test(line)) {
        const startLine = i
        
        // Check for macro (single line or continuation)
        if (trimmed.startsWith('#define')) {
          let endLine = i
          while (endLine < lines.length - 1 && lines[endLine].trimEnd().endsWith('\\')) {
            endLine++
          }
          return { startLine, endLine }
        }
        
        // Check for type alias or forward declaration (ends with ;)
        if (trimmed.includes('using') && trimmed.endsWith(';')) {
          return { startLine, endLine: i }
        }
        
        // Find opening brace
        let braceCol = line.indexOf('{')
        let braceLine = i
        
        if (braceCol === -1) {
          // Look for opening brace in subsequent lines
          for (let j = i + 1; j < Math.min(i + 30, lines.length); j++) {
            braceCol = lines[j].indexOf('{')
            if (braceCol !== -1) {
              braceLine = j
              break
            }
            // If we hit a semicolon without brace, it's a declaration
            if (lines[j].includes(';') && !lines[j].includes('{')) {
              return { startLine, endLine: j }
            }
          }
        }
        
        if (braceCol !== -1) {
          const endLine = findMatchingBrace(lines, braceLine, braceCol)
          if (endLine !== -1) {
            return { startLine, endLine }
          }
        }
        
        // Fallback: return single line
        return { startLine, endLine: i }
      }
    }
  }
  
  return null
}

/**
 * Extract source code for a symbol
 */
function extractSourceCode(
  filePath: string,
  symbolName: string,
  searchPaths: string[]
): SourceCodeResult {
  
  // Find the file
  const fullPath = findFile(filePath, searchPaths)
  if (!fullPath) {
    throw new Error(
      `[vite-source-code] File not found: "${filePath}"\n` +
      `Searched in:\n${searchPaths.map(p => `  - ${p}`).join('\n')}`
    )
  }
  
  // Read file
  const content = fs.readFileSync(fullPath, 'utf-8')
  const lines = content.split('\n')
  
  // Find symbol
  const location = findSymbolDefinition(lines, symbolName)
  if (!location) {
    throw new Error(
      `[vite-source-code] Symbol "${symbolName}" not found in "${filePath}"\n` +
      `File path: ${fullPath}`
    )
  }
  
  // Extract code
  const codeLines = lines.slice(location.startLine, location.endLine + 1)
  const code = codeLines.join('\n')
  
  return {
    code,
    file: fullPath,
    startLine: location.startLine + 1, // 1-indexed
    endLine: location.endLine + 1,
    symbol: symbolName
  }
}

// =============================================================================
// Vite Plugin
// =============================================================================

const VIRTUAL_PREFIX = 'virtual:scl-source'
const RESOLVED_PREFIX = '\0virtual:scl-source'

export function viteSourceCodePlugin(options: PluginOptions = {}): Plugin {
  let config: ResolvedConfig
  let projectRoot: string
  let searchPaths: string[]
  
  return {
    name: 'vite-scl-source-code',
    
    configResolved(resolvedConfig) {
      config = resolvedConfig
      
      // Determine project root (go up from docs/.vitepress)
      projectRoot = options.projectRoot || path.resolve(config.root, '../..')
      
      // Build search paths
      searchPaths = [
        projectRoot,
        path.join(projectRoot, 'scl'),
        path.join(projectRoot, 'scl/core'),
        path.join(projectRoot, 'scl/kernel'),
        path.join(projectRoot, 'scl/math'),
        path.join(projectRoot, 'scl/threading'),
        path.join(projectRoot, 'scl/io'),
        path.join(projectRoot, 'scl/mmap'),
        ...(options.searchPaths || [])
      ]
    },
    
    resolveId(id) {
      if (id.startsWith(VIRTUAL_PREFIX)) {
        return RESOLVED_PREFIX + id.slice(VIRTUAL_PREFIX.length)
      }
      return null
    },
    
    load(id) {
      if (!id.startsWith(RESOLVED_PREFIX)) {
        return null
      }
      
      // Parse query parameters
      const queryString = id.slice(RESOLVED_PREFIX.length)
      const params = new URLSearchParams(queryString.startsWith('?') ? queryString.slice(1) : queryString)
      
      const file = params.get('file')
      const symbol = params.get('symbol')
      
      if (!file || !symbol) {
        throw new Error(
          `[vite-source-code] Missing required parameters.\n` +
          `Usage: import data from 'virtual:scl-source?file=path/to/file.hpp&symbol=SymbolName'`
        )
      }
      
      try {
        const result = extractSourceCode(file, symbol, searchPaths)
        
        // Return as ES module
        return `
          export const code = ${JSON.stringify(result.code)};
          export const file = ${JSON.stringify(result.file)};
          export const startLine = ${result.startLine};
          export const endLine = ${result.endLine};
          export const symbol = ${JSON.stringify(result.symbol)};
          export default { code, file, startLine, endLine, symbol };
        `
      } catch (error) {
        // Re-throw with clear error message for debugging
        const message = error instanceof Error ? error.message : String(error)
        this.error(message)
      }
    }
  }
}

export default viteSourceCodePlugin

