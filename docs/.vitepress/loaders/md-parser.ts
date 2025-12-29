/**
 * Markdown Section Parser
 *
 * Parses markdown files with ## sections into a structured object.
 * Each ## heading becomes a key, content becomes the value.
 */

import type { CApiFunctionContent } from './types'

interface ParsedMarkdown {
  frontmatter: Record<string, unknown>
  sections: Record<string, string>
}

/**
 * Parse frontmatter from markdown content
 */
function parseFrontmatter(content: string): { frontmatter: Record<string, unknown>; body: string } {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/)
  if (!match) {
    return { frontmatter: {}, body: content }
  }

  const frontmatterStr = match[1]
  const body = match[2]

  // Simple YAML-like parsing for frontmatter
  const frontmatter: Record<string, unknown> = {}
  for (const line of frontmatterStr.split('\n')) {
    const colonIndex = line.indexOf(':')
    if (colonIndex > 0) {
      const key = line.slice(0, colonIndex).trim()
      const value = line.slice(colonIndex + 1).trim()
      frontmatter[key] = value
    }
  }

  return { frontmatter, body }
}

/**
 * Parse markdown body into sections based on ## headings
 */
function parseSections(body: string): Record<string, string> {
  const sections: Record<string, string> = {}
  const lines = body.split('\n')
  let currentSection = ''
  let content: string[] = []

  for (const line of lines) {
    if (line.startsWith('## ')) {
      // Save previous section
      if (currentSection) {
        sections[currentSection] = content.join('\n').trim()
      }
      // Start new section - normalize key to snake_case
      currentSection = line.slice(3).trim().toLowerCase().replace(/\s+/g, '_')
      content = []
    } else if (currentSection) {
      content.push(line)
    }
  }

  // Save last section
  if (currentSection) {
    sections[currentSection] = content.join('\n').trim()
  }

  return sections
}

/**
 * Parse a markdown file into frontmatter and sections
 */
export function parseMarkdown(content: string): ParsedMarkdown {
  const { frontmatter, body } = parseFrontmatter(content)
  const sections = parseSections(body)
  return { frontmatter, sections }
}

/**
 * Convert parsed sections to CApiFunctionContent type
 */
export function toFunctionContent(sections: Record<string, string>): CApiFunctionContent {
  return {
    brief: sections.brief || '',
    formula: sections.formula || undefined,
    description: sections.description || '',
    ffi_stability: sections.ffi_stability || '',
    future_changes: sections.future_changes || undefined,
    data_guarantees: sections.data_guarantees || '',
    mutability: sections.mutability || '',
    thread_safety: sections.thread_safety || '',
    notes: sections.notes || undefined
  }
}

/**
 * Validate that required sections are present
 */
export function validateContent(content: CApiFunctionContent, functionId: string): string[] {
  const errors: string[] = []
  const required = ['brief', 'description', 'ffi_stability', 'data_guarantees', 'mutability', 'thread_safety']

  for (const field of required) {
    if (!content[field as keyof CApiFunctionContent]) {
      errors.push(`${functionId}: Missing required section "${field}"`)
    }
  }

  return errors
}
