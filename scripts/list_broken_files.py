#!/usr/bin/env python3
"""List all broken C API files with error counts."""

import sys
import re
from collections import defaultdict
from pathlib import Path

def parse_build_log(logfile):
    """Parse build log and group errors by source file."""
    errors_by_file = defaultdict(list)
    
    with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Match error lines
            match = re.match(r'^(/[^:]+):(\d+):(?:\d+:)?\s+error:\s+(.+)$', line.rstrip())
            if match:
                filepath, line_num, message = match.groups()
                
                # Filter external dependencies
                if '/build/cmake/_deps/' in filepath:
                    continue
                if '/scl/binding/c_api/' in filepath:
                    errors_by_file[filepath].append((int(line_num), message))
    
    return errors_by_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python list_broken_files.py <build.log>")
        sys.exit(1)
    
    logfile = sys.argv[1]
    if not Path(logfile).exists():
        print(f"Error: {logfile} not found")
        sys.exit(1)
    
    errors = parse_build_log(logfile)
    
    if not errors:
        print("✓ No errors in C API files!")
        sys.exit(0)
    
    # Sort by error count
    sorted_files = sorted(errors.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("=" * 80)
    print("BROKEN C API FILES (sorted by error count)")
    print("=" * 80)
    print()
    
    for filepath, file_errors in sorted_files:
        # Extract relative path
        if '/scl/' in filepath:
            rel_path = filepath[filepath.index('/scl/'):]
        else:
            rel_path = filepath
        
        print(f"📄 {rel_path}")
        print(f"   {len(file_errors)} errors")
        
        # Show first 3 unique error types
        unique_errors = {}
        for line_num, msg in file_errors[:10]:
            short_msg = msg[:80]
            if short_msg not in unique_errors:
                unique_errors[short_msg] = (line_num, msg)
        
        for i, (line_num, msg) in enumerate(list(unique_errors.values())[:3]):
            print(f"   • Line {line_num}: {msg[:70]}...")
        
        if len(file_errors) > 3:
            print(f"   ... and {len(file_errors) - 3} more errors")
        print()
    
    # Suggest which file to fix first
    print("=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    first_file = sorted_files[0][0]
    if '/scl/' in first_file:
        first_file = first_file[first_file.index('/scl/'):]
    print(f"Fix this file first: {first_file}")
    print(f"Command: python scripts/fix_one_file.py {first_file}")

if __name__ == '__main__':
    main()

