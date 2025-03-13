#!/usr/bin/env python3
"""
Fix script for sitecustomize.py recursion issues.
This script replaces the problematic sitecustomize.py with a fixed version.
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

def fix_sitecustomize():
    """Fix the sitecustomize.py file to prevent recursion."""
    # Get site-packages directory
    site_packages = site.getsitepackages()[0]
    
    # Path to sitecustomize.py
    sitecustomize_path = os.path.join(site_packages, 'sitecustomize.py')
    
    # Check if file exists
    if not os.path.exists(sitecustomize_path):
        print_error(f"sitecustomize.py not found at {sitecustomize_path}")
        return False
    
    # Create backup
    backup_path = sitecustomize_path + '.bak'
    if not os.path.exists(backup_path):
        print_info(f"Creating backup of sitecustomize.py at {backup_path}")
        shutil.copy2(sitecustomize_path, backup_path)
        print_success(f"Backup created at {backup_path}")
    else:
        print_warning(f"Backup already exists at {backup_path}")
    
    # Create fixed sitecustomize.py
    print_info(f"Creating fixed sitecustomize.py at {sitecustomize_path}")
    with open(sitecustomize_path, 'w') as f:
        f.write("""
# Fixed sitecustomize.py to prevent recursion issues
import sys
import os
import importlib.util
import importlib.machinery

# Print debug info
print("sitecustomize.py loaded", file=sys.stderr)

# Set environment variables to prevent pip from installing dependencies
os.environ["PIP_NO_DEPENDENCIES"] = "1"
os.environ["PIP_NO_INSTALL"] = "1"

# Create a custom import hook to prevent automatic installation
class PreventAutoInstallImportHook:
    def __init__(self):
        self.blocked_packages = ['torch', 'torchvision', 'torchaudio']
        # Keep track of packages we've already tried to find to prevent recursion
        self.checked_packages = set()
    
    def find_spec(self, fullname, path, target=None):
        # Prevent recursion by checking if we've already tried to find this package
        if fullname in self.checked_packages:
            return None
        
        # Check if this is a package we want to block
        if any(fullname.startswith(pkg) for pkg in self.blocked_packages):
            # Add to checked packages to prevent recursion
            self.checked_packages.add(fullname)
            
            # Try to find the package directly using the loader
            try:
                # Try to find the module directly
                loader = importlib.machinery.PathFinder.find_spec(fullname, path)
                if loader is not None:
                    return loader
            except Exception:
                pass
            
            # If not found, print a warning and return None
            print(f"WARNING: Blocked automatic installation of {fullname}", file=sys.stderr)
            return None
        
        # Return None to let the normal import system handle it
        return None

# Register the import hook
sys.meta_path.insert(0, PreventAutoInstallImportHook())
""")
    
    print_success(f"Fixed sitecustomize.py created at {sitecustomize_path}")
    return True

def main():
    """Main function."""
    print_info("Fixing sitecustomize.py to prevent recursion issues")
    
    if fix_sitecustomize():
        print_success("sitecustomize.py fixed successfully")
    else:
        print_error("Failed to fix sitecustomize.py")
        sys.exit(1)

if __name__ == "__main__":
    main()