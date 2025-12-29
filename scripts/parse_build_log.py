#!/usr/bin/env python3
"""
Parse C++ build logs and present errors/warnings in a structured format.

Usage:
    python scripts/parse_build_log.py [build.log]
    make build 2>&1 | python scripts/parse_build_log.py
    python scripts/parse_build_log.py build.log --html report.html
    python scripts/parse_build_log.py build.log --verbose
"""

import sys
import re
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from pathlib import Path


@dataclass
class CompileIssue:
    """Represents a single compilation error or warning."""
    file: str
    line: int
    severity: str  # 'error' or 'warning'
    message: str
    context: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.file, self.line, self.severity, self.message))
    
    @property
    def short_file(self) -> str:
        """Get short file path starting from /scl/."""
        if '/scl/' in self.file:
            return self.file[self.file.index('/scl/'):]
        return self.file


class BuildLogParser:
    """Parser for C++ build logs."""
    
    # Error patterns to identify root causes
    ERROR_PATTERNS = {
        'function_call_mismatch': r'no matching function for call',
        'missing_member': r'has no member named',
        'undeclared_identifier': r'(does not name a type|was not declared)',
        'type_conversion': r'cannot convert',
        'static_assert_failure': r'static assertion failed',
        'syntax_error': r'expected .+ before',
        'template_error': r'template argument deduction',
        'constraint_not_satisfied': r'constraints not satisfied',
    }
    
    def __init__(self):
        self.errors: List[CompileIssue] = []
        self.warnings: List[CompileIssue] = []
        self.unique_warnings: Dict[str, Set[str]] = defaultdict(set)
        self.failed_targets: List[str] = []
        self.cmake_warnings: List[str] = []
        
    def parse_file(self, filepath: str):
        """Parse a build log file."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            self.parse_lines(f)
    
    def parse_stdin(self):
        """Parse build log from stdin."""
        self.parse_lines(sys.stdin)
    
    def parse_lines(self, lines):
        """Parse build log lines."""
        current_issue = None
        context_buffer = []
        
        for line in lines:
            line = line.rstrip()
            
            # Track CMake warnings
            if line.startswith('CMake Warning') or line.startswith('CMake Deprecation Warning'):
                self.cmake_warnings.append(line)
            
            # Track failed targets
            if 'gmake[3]: ***' in line and 'Error' in line:
                match = re.search(r'CMakeFiles/[^/]+/([^/]+\.cpp\.o)', line)
                if match:
                    self.failed_targets.append(match.group(1))
            
            # Match error/warning lines
            # Format: /path/to/file.cpp:123:45: error: message
            match = re.match(
                r'^(/[^:]+):(\d+):(?:\d+:)?\s+(error|warning):\s+(.+)$',
                line
            )
            
            if match:
                filepath, line_num, severity, message = match.groups()
                
                # Filter out pedantic warnings from external dependencies
                if 'pedantic' in message.lower() and 'hwy' in filepath:
                    continue
                
                # Filter out attribute ignored warnings (known issue)
                if "'maybe_unused' attribute ignored" in message:
                    self.unique_warnings['maybe_unused'].add(filepath)
                    continue
                
                # Filter out terminate warnings
                if "'throw' will always call 'terminate'" in message:
                    self.unique_warnings['terminate_warning'].add(filepath)
                    continue
                
                issue = CompileIssue(
                    file=filepath,
                    line=int(line_num),
                    severity=severity,
                    message=message,
                    context=context_buffer[-3:] if context_buffer else []
                )
                
                if severity == 'error':
                    self.errors.append(issue)
                else:
                    self.warnings.append(issue)
                
                current_issue = issue
                context_buffer = []
            
            # Collect context lines (notes, required from, etc.)
            elif current_issue and (
                line.strip().startswith('note:') or 
                'required from' in line or
                'In instantiation of' in line
            ):
                current_issue.context.append(line.strip())
            else:
                if current_issue:
                    current_issue = None
                context_buffer.append(line)
    
    def categorize_error(self, error: CompileIssue) -> str:
        """Categorize an error by pattern matching."""
        msg = error.message.lower()
        
        for category, pattern in self.ERROR_PATTERNS.items():
            if re.search(pattern, msg, re.IGNORECASE):
                return category
        
        return 'other'
    
    def group_errors_by_type(self) -> Dict[str, List[CompileIssue]]:
        """Group errors by their root cause."""
        groups = defaultdict(list)
        for error in self.errors:
            category = self.categorize_error(error)
            groups[category].append(error)
        return groups
    
    def extract_affected_files(self) -> Dict[str, int]:
        """Extract files with errors and count."""
        file_errors = defaultdict(int)
        for error in self.errors:
            src_file = error.short_file
            file_errors[src_file] += 1
        return dict(sorted(file_errors.items(), key=lambda x: -x[1]))
    
    def get_root_cause_suggestions(self) -> List[str]:
        """Analyze errors and suggest root causes."""
        suggestions = []
        
        # Check for common patterns
        has_member_issues = sum(1 for e in self.errors if 'has no member named' in e.message.lower())
        has_type_issues = sum(1 for e in self.errors if 'does not name a type' in e.message.lower())
        has_call_issues = sum(1 for e in self.errors if 'no matching function' in e.message.lower())
        
        if has_member_issues > 20:
            suggestions.append("⚠ High number of 'has no member' errors - possible API signature mismatch")
        
        if has_type_issues > 20:
            suggestions.append("⚠ High number of 'does not name a type' errors - missing includes or namespace issues")
        
        if has_call_issues > 50:
            suggestions.append("⚠ High number of function call mismatches - check function signatures and parameter types")
        
        # Check for specific issues
        for error in self.errors[:50]:
            if 'SparseWrapper' in error.message and 'does not name a type' in error.message:
                suggestions.append("✗ 'SparseWrapper' not found - check scl/binding/c_api/core/internal.hpp includes")
                break
        
        for error in self.errors[:50]:
            if 'zero_all' in error.message and 'WorkspacePool' in error.message:
                suggestions.append("✗ 'WorkspacePool::zero_all()' missing - check scl/threading/*.hpp API")
                break
        
        for error in self.errors[:50]:
            if 'row_indices_unsafe()' in error.message and 'candidate expects 1 argument, 0 provided' in '\n'.join(error.context):
                suggestions.append("✗ 'row_indices_unsafe()' called without argument - API changed to require index parameter")
                break
        
        for error in self.errors[:50]:
            if "has no member named 'values'" in error.message:
                suggestions.append("✗ 'Sparse::values()' member access failed - API may have changed to values_unsafe() or removed array accessor")
                break
        
        return suggestions
    
    def print_summary(self):
        """Print a concise summary of build issues."""
        print("=" * 80)
        print("🔨 BUILD LOG ANALYSIS")
        print("=" * 80)
        print()
        
        # Overall stats
        print(f"📊 Statistics:")
        print(f"   Errors:          {len(self.errors)}")
        print(f"   Warnings:        {len(self.warnings)}")
        print(f"   Failed Targets:  {len(set(self.failed_targets))}")
        print(f"   CMake Warnings:  {len(self.cmake_warnings)}")
        print()
        
        # Root cause analysis
        suggestions = self.get_root_cause_suggestions()
        if suggestions:
            print("-" * 80)
            print("🔍 ROOT CAUSE ANALYSIS")
            print("-" * 80)
            for suggestion in suggestions:
                print(f"  {suggestion}")
            print()
        
        # Suppressed warnings
        if self.unique_warnings:
            print("-" * 80)
            print("✓ SUPPRESSED WARNINGS (known/filtered)")
            print("-" * 80)
            for warning_type, files in self.unique_warnings.items():
                print(f"  {warning_type}: {len(files)} files")
            print()
        
        # Affected files
        affected = self.extract_affected_files()
        if affected:
            print("-" * 80)
            print("📁 MOST AFFECTED FILES")
            print("-" * 80)
            for file, count in list(affected.items())[:10]:
                print(f"  {count:4d} errors  {file}")
            print()
        
        # Group errors by type
        groups = self.group_errors_by_type()
        print("-" * 80)
        print("🏷️  ERROR CATEGORIES")
        print("-" * 80)
        
        category_names = {
            'undeclared_identifier': '🔴 Undeclared identifiers',
            'missing_member': '🔴 Missing member access',
            'function_call_mismatch': '🔴 Function call mismatches',
            'type_conversion': '🟠 Type conversion errors',
            'static_assert_failure': '🟠 Static assertion failures',
            'constraint_not_satisfied': '🟠 Constraint failures',
            'template_error': '🟡 Template errors',
            'syntax_error': '🟡 Syntax errors',
            'other': '⚪ Other errors'
        }
        
        for key, name in category_names.items():
            if key in groups and groups[key]:
                print(f"\n{name}: {len(groups[key])} errors")
                
                # Show unique error patterns
                unique_messages = {}
                for error in groups[key]:
                    msg = error.message[:100]
                    if msg not in unique_messages:
                        unique_messages[msg] = error
                
                for msg, error in list(unique_messages.items())[:3]:
                    print(f"  📄 {error.short_file}:{error.line}")
                    print(f"     {error.message}")
                
                if len(unique_messages) > 3:
                    print(f"  ... and {len(unique_messages) - 3} more unique patterns")
    
    def print_detailed_errors(self, max_errors: int = 20, verbose: bool = False):
        """Print detailed error information."""
        print()
        print("=" * 80)
        print("📋 DETAILED ERROR ANALYSIS")
        print("=" * 80)
        print()
        
        groups = self.group_errors_by_type()
        
        # Prioritize critical errors
        priority_order = [
            'undeclared_identifier',
            'missing_member',
            'function_call_mismatch',
            'type_conversion',
            'static_assert_failure',
            'constraint_not_satisfied',
            'template_error',
            'syntax_error',
            'other'
        ]
        
        error_count = 0
        for category in priority_order:
            if category not in groups or not groups[category]:
                continue
            
            # Group by file
            by_file = defaultdict(list)
            for error in groups[category]:
                by_file[error.short_file].append(error)
            
            for file, file_errors in list(by_file.items())[:3]:
                if error_count >= max_errors:
                    break
                
                print(f"\n{'─' * 80}")
                print(f"📄 File: {file}")
                print(f"🏷️  Category: {category.replace('_', ' ').title()}")
                print(f"📊 Count: {len(file_errors)} errors")
                print('─' * 80)
                
                # Show first few errors from this file
                show_count = 3 if verbose else 2
                for error in file_errors[:show_count]:
                    print(f"\n📍 Line {error.line}: {error.severity.upper()}")
                    print(f"   {error.message}")
                    
                    # Show relevant context
                    if verbose and error.context:
                        print("\n   Context:")
                        for ctx_line in error.context[:5]:
                            if any(kw in ctx_line for kw in ['note:', 'required from', 'candidate']):
                                print(f"   {ctx_line[:100]}")
                    
                    error_count += 1
                    if error_count >= max_errors:
                        break
                
                if len(file_errors) > show_count:
                    print(f"\n   ... and {len(file_errors) - show_count} more errors in this file")
        
        if error_count < len(self.errors):
            print(f"\n\n[Showing {error_count} of {len(self.errors)} total errors]")
            print("💡 Tip: Use --verbose or increase --max-errors to see more details")
    
    def export_html(self, output_path: str):
        """Export analysis to HTML report."""
        groups = self.group_errors_by_type()
        affected = self.extract_affected_files()
        suggestions = self.get_root_cause_suggestions()
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Build Log Analysis</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #555; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #666; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #e74c3c; }}
        .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
        .error {{ background: #ffe6e6; padding: 15px; margin: 10px 0; border-left: 4px solid #e74c3c; border-radius: 4px; }}
        .warning {{ background: #fff4e6; padding: 15px; margin: 10px 0; border-left: 4px solid #f39c12; border-radius: 4px; }}
        .suggestion {{ background: #e8f4f8; padding: 12px; margin: 8px 0; border-left: 4px solid #3498db; border-radius: 4px; }}
        .file-path {{ font-family: 'Courier New', monospace; color: #2c3e50; font-weight: bold; }}
        .error-msg {{ color: #c0392b; font-family: 'Courier New', monospace; font-size: 13px; }}
        .context {{ color: #7f8c8d; font-family: 'Courier New', monospace; font-size: 11px; margin-left: 20px; margin-top: 5px; }}
        .category-badge {{ display: inline-block; padding: 4px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; margin-right: 8px; }}
        .badge-error {{ background: #e74c3c; color: white; }}
        .badge-warning {{ background: #f39c12; color: white; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
        tr:hover {{ background: #f9f9f9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔨 Build Log Analysis Report</h1>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{len(self.errors)}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(self.warnings)}</div>
                <div class="stat-label">Warnings</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(set(self.failed_targets))}</div>
                <div class="stat-label">Failed Targets</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(affected)}</div>
                <div class="stat-label">Affected Files</div>
            </div>
        </div>
        
        <h2>🔍 Root Cause Analysis</h2>
        {''.join(f'<div class="suggestion">{s}</div>' for s in suggestions) if suggestions else '<p>No specific suggestions available.</p>'}
        
        <h2>📁 Most Affected Files</h2>
        <table>
            <tr><th>Error Count</th><th>File</th></tr>
            {''.join(f'<tr><td>{count}</td><td class="file-path">{file}</td></tr>' for file, count in list(affected.items())[:15])}
        </table>
        
        <h2>🏷️ Error Categories</h2>
"""
        
        category_names = {
            'undeclared_identifier': 'Undeclared Identifiers',
            'missing_member': 'Missing Member Access',
            'function_call_mismatch': 'Function Call Mismatches',
            'type_conversion': 'Type Conversion Errors',
            'static_assert_failure': 'Static Assertion Failures',
            'constraint_not_satisfied': 'Constraint Failures',
            'template_error': 'Template Errors',
            'syntax_error': 'Syntax Errors',
            'other': 'Other Errors'
        }
        
        for key, name in category_names.items():
            if key in groups and groups[key]:
                html += f"<h3>{name} ({len(groups[key])} errors)</h3>\n"
                
                # Show sample errors
                for error in groups[key][:5]:
                    html += f"""<div class="error">
    <div><span class="category-badge badge-error">ERROR</span><span class="file-path">{error.short_file}:{error.line}</span></div>
    <div class="error-msg">{error.message}</div>
"""
                    if error.context:
                        html += '<div class="context">' + '<br>'.join(error.context[:3]) + '</div>'
                    html += "</div>\n"
                
                if len(groups[key]) > 5:
                    html += f"<p><em>... and {len(groups[key]) - 5} more errors</em></p>\n"
        
        html += """
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"\n✓ HTML report exported to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Parse C++ build logs and present errors in structured format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s build.log
  %(prog)s build.log --html report.html
  %(prog)s build.log --verbose --max-errors 30
  make build 2>&1 | %(prog)s
        """
    )
    
    parser.add_argument(
        'logfile',
        nargs='?',
        help='Build log file to parse (or read from stdin)'
    )
    parser.add_argument(
        '--html',
        metavar='FILE',
        help='Export analysis to HTML report'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show more context for each error'
    )
    parser.add_argument(
        '--max-errors',
        type=int,
        default=15,
        help='Maximum number of detailed errors to show (default: 15)'
    )
    
    args = parser.parse_args()
    
    log_parser = BuildLogParser()
    
    if args.logfile:
        # Read from file
        if not Path(args.logfile).exists():
            print(f"Error: File '{args.logfile}' not found", file=sys.stderr)
            sys.exit(1)
        log_parser.parse_file(args.logfile)
    else:
        # Read from stdin
        if sys.stdin.isatty():
            parser.print_help()
            sys.exit(1)
        log_parser.parse_stdin()
    
    # Print results
    log_parser.print_summary()
    log_parser.print_detailed_errors(max_errors=args.max_errors, verbose=args.verbose)
    
    # Export HTML if requested
    if args.html:
        log_parser.export_html(args.html)
    
    print("\n" + "=" * 80)
    if log_parser.errors:
        print(f"❌ Build failed with {len(log_parser.errors)} errors")
        sys.exit(1)
    else:
        print("✓ Build completed successfully (warnings only)")
        sys.exit(0)


if __name__ == '__main__':
    main()
