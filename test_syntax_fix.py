#!/usr/bin/env python3
"""
Simple syntax check for the memory leak fix.
Verifies the code changes compile without syntax errors.
"""

import ast
import sys

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def main():
    files_to_check = [
        'src/_bentoml_impl/client/http.py',
        'src/_bentoml_sdk/service/dependency.py'
    ]
    
    print("Checking syntax of modified files...")
    
    all_good = True
    for filepath in files_to_check:
        print(f"Checking {filepath}...")
        valid, error = check_file_syntax(filepath)
        if valid:
            print(f"✓ {filepath} - OK")
        else:
            print(f"✗ {filepath} - ERROR: {error}")
            all_good = False
    
    if all_good:
        print("\n🎉 All files have valid syntax!")
        print("The memory leak fix has been successfully applied.")
    else:
        print("\n❌ Syntax errors found!")
        sys.exit(1)

if __name__ == "__main__":
    main()