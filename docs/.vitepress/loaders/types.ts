/**
 * C API Documentation Type Definitions
 *
 * This file defines the TypeScript interfaces for the data/content separation architecture.
 * - YAML data (Codegen generated): Structured metadata
 * - Markdown content (AI/Human written): Descriptive text
 */

// =============================================================================
// YAML Data Types (Codegen Generated)
// =============================================================================

export interface CApiModule {
  name: string
  header: string
  description: string
  functions: CApiFunctionData[]
}

export interface CApiFunctionData {
  id: string
  return_type: string
  source: {
    file: string
    line: number
  }
  params: CApiParam[]
  errors: CApiError[]
  complexity?: {
    time: string
    space: string
  }
  version: string
  status: 'stable' | 'beta' | 'experimental' | 'deprecated'
}

export interface CApiParam {
  name: string
  type: string
  dir: 'in' | 'out' | 'inout'
  nullable?: boolean
  default?: string
}

export interface CApiError {
  code: string
  condition: string
}

// =============================================================================
// Markdown Content Types (AI/Human Written)
// =============================================================================

export interface CApiFunctionContent {
  brief: string
  formula?: string
  description: string
  ffi_stability: string
  future_changes?: string
  data_guarantees: string
  mutability: string
  thread_safety: string
  notes?: string
}

// =============================================================================
// Merged Function Type (Data + Content)
// =============================================================================

export interface CApiFunction extends CApiFunctionData {
  content: CApiFunctionContent
}

export interface CApiModuleWithContent {
  name: string
  header: string
  description: string
  functions: CApiFunction[]
}

// =============================================================================
// Loader Output Type
// =============================================================================

export interface CApiData {
  modules: CApiModuleWithContent[]
  functionIndex: Map<string, CApiFunction>
}
