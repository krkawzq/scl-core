/**
 * VitePress Markdown Plugin for ::: source_code blocks
 * 
 * This plugin extracts C++ code at BUILD TIME and outputs standard markdown
 * code blocks that VitePress will render with its built-in features:
 * - Syntax highlighting (Shiki)
 * - Line numbers
 * - Copy button
 * - Horizontal scrolling
 * 
 * Usage:
 *   ::: source_code file="scl/core/memory.hpp" symbol="aligned_alloc"
 *   :::
 */

import type MarkdownIt from 'markdown-it'
import container from 'markdown-it-container'
import fs from 'fs'
import path from 'path'

interface PluginOptions {
  projectRoot: string
}

// =============================================================================
// C++ Parser Functions
// =============================================================================

function findFile(filename: string, searchPaths: string[]): string | null {
  for (const searchPath of searchPaths) {
    const fullPath = path.join(searchPath, filename)
    if (fs.existsSync(fullPath)) {
      return fullPath
    }
  }
  return null
}

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
        if (char === '/' && nextChar === '/') {
          break
        }
      }
      
      if (inBlockComment) continue
      
      if (char === '"' && prevChar !== '\\') {
        inString = !inString
        continue
      }
      if (char === "'" && prevChar !== '\\') {
        inChar = !inChar
        continue
      }
      
      if (inString || inChar) continue
      
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

function findSymbolDefinition(lines: string[], symbolName: string): { startLine: number; endLine: number } | null {
  const escapedSymbol = symbolName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  
  const patterns: RegExp[] = [
    new RegExp(`^\\s*template\\s*<[^>]*>\\s*(?:inline\\s+)?(?:static\\s+)?(?:constexpr\\s+)?(?:[\\w:]+\\s+)*${escapedSymbol}\\s*[<(]`),
    new RegExp(`^\\s*(?:inline\\s+)?(?:static\\s+)?(?:constexpr\\s+)?(?:SCL_\\w+\\s+)?(?:[\\w:*&<>]+\\s+)+${escapedSymbol}\\s*\\(`),
    new RegExp(`^\\s*(?:template\\s*<[^>]*>\\s*)?(?:class|struct)\\s+${escapedSymbol}\\s*[:{<]`),
    new RegExp(`^\\s*enum\\s+(?:class\\s+)?${escapedSymbol}\\s*[:{]`),
    new RegExp(`^\\s*using\\s+${escapedSymbol}\\s*=`),
    new RegExp(`^\\s*#define\\s+${escapedSymbol}[\\s(]`),
  ]
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    
    if (trimmed.startsWith('//') || trimmed.startsWith('/*') || trimmed.startsWith('*')) {
      continue
    }
    
    for (const pattern of patterns) {
      if (pattern.test(line)) {
        const startLine = i
        
        if (trimmed.startsWith('#define')) {
          let endLine = i
          while (endLine < lines.length - 1 && lines[endLine].trimEnd().endsWith('\\')) {
            endLine++
          }
          return { startLine, endLine }
        }
        
        if (trimmed.includes('using') && trimmed.endsWith(';')) {
          return { startLine, endLine: i }
        }
        
        let braceCol = line.indexOf('{')
        let braceLine = i
        
        if (braceCol === -1) {
          for (let j = i + 1; j < Math.min(i + 30, lines.length); j++) {
            braceCol = lines[j].indexOf('{')
            if (braceCol !== -1) {
              braceLine = j
              break
            }
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
        
        return { startLine, endLine: i }
      }
    }
  }
  
  return null
}

interface ExtractResult {
  code: string
  file: string
  relativePath: string
  startLine: number
  endLine: number
}

function extractSourceCode(filePath: string, symbolName: string, projectRoot: string): ExtractResult {
  const searchPaths = [
    projectRoot,
    path.join(projectRoot, 'scl'),
    path.join(projectRoot, 'scl/core'),
    path.join(projectRoot, 'scl/kernel'),
    path.join(projectRoot, 'scl/math'),
    path.join(projectRoot, 'scl/threading'),
    path.join(projectRoot, 'scl/io'),
    path.join(projectRoot, 'scl/mmap'),
  ]
  
  const fullPath = findFile(filePath, searchPaths)
  if (!fullPath) {
    throw new Error(`File not found: "${filePath}"\nSearched in:\n${searchPaths.map(p => `  - ${p}`).join('\n')}`)
  }
  
  const content = fs.readFileSync(fullPath, 'utf-8')
  const lines = content.split('\n')
  
  const location = findSymbolDefinition(lines, symbolName)
  if (!location) {
    throw new Error(`Symbol "${symbolName}" not found in "${filePath}"`)
  }
  
  const codeLines = lines.slice(location.startLine, location.endLine + 1)
  
  // Get relative path for display
  const parts = fullPath.split(/[/\\]/)
  const sclIndex = parts.findIndex(p => p === 'scl')
  const relativePath = sclIndex !== -1 
    ? parts.slice(sclIndex).join('/') 
    : parts.slice(-3).join('/')
  
  return {
    code: codeLines.join('\n'),
    file: fullPath,
    relativePath,
    startLine: location.startLine + 1,
    endLine: location.endLine + 1
  }
}

// =============================================================================
// Parameter Parser
// =============================================================================

interface SourceCodeParams {
  file: string
  symbol: string
  title?: string
  lang?: string
  collapsed?: boolean
}

function parseParams(paramString: string): SourceCodeParams {
  const params: SourceCodeParams = { file: '', symbol: '', collapsed: false }
  
  // Check for 'collapsed' flag (no value needed)
  if (paramString.includes('collapsed')) {
    params.collapsed = true
  }
  
  const regex = /(\w+)="([^"]*)"/g
  let match
  while ((match = regex.exec(paramString)) !== null) {
    const [, key, value] = match
    if (key === 'file') params.file = value
    else if (key === 'symbol') params.symbol = value
    else if (key === 'title') params.title = value
    else if (key === 'lang') params.lang = value
  }
  
  return params
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

// =============================================================================
// Markdown Plugin - Output tokens for VitePress to render
// =============================================================================

export function sourceCodePlugin(md: MarkdownIt, options: PluginOptions) {
  const projectRoot = options.projectRoot
  
  md.use(container, 'source_code', {
    validate: (params: string) => {
      return params.trim().match(/^source_code\s+/)
    },
    
    render: (tokens: any[], idx: number, _options: any, env: any, self: any) => {
      const token = tokens[idx]
      
      if (token.nesting === 1) {
        const paramString = token.info.trim().slice('source_code'.length).trim()
        const params = parseParams(paramString)
        
        if (!params.file || !params.symbol) {
          return `<div class="custom-block danger"><p class="custom-block-title">Error</p><p>Missing required parameters: file and symbol</p></div>\n`
        }
        
        try {
          const result = extractSourceCode(params.file, params.symbol, projectRoot)
          const title = params.title || params.symbol
          const lang = params.lang || 'cpp'
          
          // Output CollapsibleCode component as Vue component tag
          // Pass raw code string, component will render it as code block
          const collapsedAttr = params.collapsed ? ' collapsed' : ''
          const titleAttr = params.title ? ` title="${escapeHtml(params.title)}"` : ''
          
          // Escape the raw code for use in Vue template string
          const escapedCode = result.code
            .replace(/\\/g, '\\\\')
            .replace(/`/g, '\\`')
            .replace(/\$/g, '\\$')
          
          const html = `<CollapsibleCode
  file="${escapeHtml(result.file)}"
  symbol="${escapeHtml(params.symbol)}"
  code="${escapeHtml(escapedCode)}"
  start-line="${result.startLine}"
  end-line="${result.endLine}"
  lang="${lang}"${titleAttr}${collapsedAttr}
/>\n`
          
          return html
          
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error)
          console.error(`[source_code] Error:`, message)
          
          return `<div class="custom-block danger">
<p class="custom-block-title">Source Code Error</p>
<p><code>${escapeHtml(message)}</code></p>
</div>\n`
        }
        
      } else {
        // Closing tag - return empty (content already rendered in opening)
        return ''
      }
    }
  })
}

export default sourceCodePlugin
