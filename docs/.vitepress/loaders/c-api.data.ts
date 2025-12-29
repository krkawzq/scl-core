/**
 * C API Data Loader
 *
 * VitePress data loader that:
 * 1. Loads YAML files from _data/ (Codegen generated)
 * 2. Loads MD files from module directories (AI/Human written)
 * 3. Merges them into a unified data structure
 */

import { defineLoader } from 'vitepress'
import { parse as parseYaml } from 'yaml'
import { readFileSync, readdirSync, existsSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { parseMarkdown, toFunctionContent, validateContent } from './md-parser'
import type { CApiModule, CApiModuleWithContent, CApiFunction, CApiData } from './types'

const __dirname = dirname(fileURLToPath(import.meta.url))
const API_DIR = join(__dirname, '../../api/c-api')
const DATA_DIR = join(API_DIR, '_data')

export type { CApiData, CApiFunction, CApiModuleWithContent }

export default defineLoader({
  watch: [
    join(DATA_DIR, '*.yaml'),
    join(API_DIR, '**/*.md')
  ],
  async load(): Promise<CApiData> {
    const modules: CApiModuleWithContent[] = []
    const functionIndex = new Map<string, CApiFunction>()
    const warnings: string[] = []

    // Check if data directory exists
    if (!existsSync(DATA_DIR)) {
      console.warn('[c-api loader] Data directory not found:', DATA_DIR)
      return { modules, functionIndex }
    }

    // Load all YAML data files
    const dataFiles = readdirSync(DATA_DIR).filter(f => f.endsWith('.yaml'))

    for (const file of dataFiles) {
      try {
        const yamlPath = join(DATA_DIR, file)
        const yamlContent = readFileSync(yamlPath, 'utf-8')
        const moduleData: CApiModule = parseYaml(yamlContent)

        const functionsWithContent: CApiFunction[] = []

        // Load corresponding MD content for each function
        for (const funcData of moduleData.functions) {
          // Derive MD filename from function ID
          // scl_algebra_spmv -> algebra/spmv.md
          const funcName = funcData.id.replace(`scl_${moduleData.name}_`, '')
          const mdPath = join(API_DIR, moduleData.name, `${funcName}.md`)

          let content = {
            brief: '',
            description: '',
            ffi_stability: '',
            data_guarantees: '',
            mutability: '',
            thread_safety: ''
          }

          if (existsSync(mdPath)) {
            const mdContent = readFileSync(mdPath, 'utf-8')
            const parsed = parseMarkdown(mdContent)
            content = toFunctionContent(parsed.sections)

            // Validate content
            const errors = validateContent(content, funcData.id)
            if (errors.length > 0) {
              warnings.push(...errors)
            }
          } else {
            warnings.push(`${funcData.id}: MD file not found at ${mdPath}`)
          }

          const funcWithContent: CApiFunction = {
            ...funcData,
            content
          }

          functionsWithContent.push(funcWithContent)
          functionIndex.set(funcData.id, funcWithContent)
        }

        modules.push({
          name: moduleData.name,
          header: moduleData.header,
          description: moduleData.description,
          functions: functionsWithContent
        })
      } catch (err) {
        console.error(`[c-api loader] Error loading ${file}:`, err)
      }
    }

    // Log warnings in development
    if (warnings.length > 0 && process.env.NODE_ENV !== 'production') {
      console.warn('[c-api loader] Warnings:')
      warnings.forEach(w => console.warn('  -', w))
    }

    return { modules, functionIndex }
  }
})
