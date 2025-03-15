#!/usr/bin/env python3
"""
Fix script for readline issues in the Python interactive shell.
This script updates the sitecustomize.py file to handle readline properly.
"""
import os
import sys
import site
import shutil

def print_info(text):
    """Print formatted info text."""
    print(f"[INFO] {text}")

def print_error(text):
    """Print formatted error text."""
    print(f"[ERROR] {text}")

def print_success(text):
    """Print formatted success text."""
    print(f"[SUCCESS] {text}")

def print_warning(text):
    """Print formatted warning text."""
    print(f"[WARNING] {text}")

def fix_sitecustomize_readline():
    """Fix the sitecustomize.py file to handle readline properly."""
    # Get site-packages directory
    site_packages = site.getsitepackages()[0]
    
    # Path to sitecustomize.py
    sitecustomize_path = os.path.join(site_packages, 'sitecustomize.py')
    
    # Check if file exists
    if not os.path.exists(sitecustomize_path):
        print_error(f"sitecustomize.py not found at {sitecustomize_path}")
        return False
    
    # Create backup if it doesn't exist
    backup_path = sitecustomize_path + '.readline.bak'
    if not os.path.exists(backup_path):
        print_info(f"Creating backup of sitecustomize.py at {backup_path}")
        shutil.copy2(sitecustomize_path, backup_path)
        print_success(f"Backup created at {backup_path}")
    else:
        print_warning(f"Backup already exists at {backup_path}")
    
    # Read the current content
    with open(sitecustomize_path, 'r') as f:
        content = f.read()
    
    # Add readline fix
    readline_fix = """
# Fix for readline module in interactive shell
try:
    import readline
    # Check if we're in interactive mode
    if hasattr(sys, 'ps1'):
        try:
            # Only call register_readline if it exists and we're in interactive mode
            if hasattr(sys, '__interactivehook__'):
                # Patch readline.backend if it doesn't exist
                if not hasattr(readline, 'backend'):
                    readline.backend = 'readline'
        except Exception as e:
            print(f"Warning: Readline initialization error: {e}", file=sys.stderr)
except ImportError:
    # Readline not available, skip
    pass
"""
    
    # Check if the fix is already in the file
    if "Fix for readline module in interactive shell" in content:
        print_info("Readline fix already present in sitecustomize.py")
        return True
    
    # Add the fix at the beginning of the file
    new_content = readline_fix + content
    
    # Write the updated content
    with open(sitecustomize_path, 'w') as f:
        f.write(new_content)
    
    print_success(f"Added readline fix to {sitecustomize_path}")
    return True

def main():
    """Main function."""
    print_info("Fixing sitecustomize.py to handle readline properly")
    
    if fix_sitecustomize_readline():
        print_success("sitecustomize.py fixed successfully for readline")
    else:
        print_error("Failed to fix sitecustomize.py for readline")
        sys.exit(1)

if __name__ == "__main__":
    main()