#!/usr/bin/env python3
"""
Verify PyTorch installation and functionality.
This script attempts to import PyTorch and run basic operations.
"""
import os
import sys

# Disable sitecustomize.py and other import hooks to prevent recursion issues
os.environ["PYTHONNOUSERSITE"] = "1"  # Disable user site-packages
os.environ["PYTHONPATH"] = ""  # Clear PYTHONPATH

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

def verify_torch():
    """Verify PyTorch installation and functionality."""
    print_info("Verifying PyTorch installation")
    
    # Add site-packages to sys.path
    site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
    
    # Print sys.path for debugging
    print_info("Python path:")
    for path in sys.path:
        print(f"  - {path}")
    
    # Try to import torch
    try:
        print_info("Attempting to import torch")
        import torch
        print_success(f"PyTorch is installed (version {torch.__version__})")
        print_info(f"PyTorch location: {torch.__file__}")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print_success(f"CUDA is available (version {torch.version.cuda})")
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
            
            # Test a simple CUDA operation
            try:
                x = torch.rand(5, 3).cuda()
                y = torch.rand(5, 3).cuda()
                z = x + y
                print_success("Basic CUDA tensor operations work correctly")
            except Exception as e:
                print_warning(f"CUDA tensor operations failed: {e}")
                print_warning("Falling back to CPU mode")
        else:
            print_info("CUDA is not available, using CPU-only mode")
        
        # Test a simple tensor operation
        try:
            x = torch.rand(5, 3)
            y = torch.rand(5, 3)
            z = x + y
            print_success("Basic tensor operations work correctly")
        except Exception as e:
            print_error(f"Failed to perform basic tensor operations: {e}")
            return False
        
        return True
    except ImportError as e:
        print_error(f"PyTorch is not installed: {e}")
        return False
    except Exception as e:
        print_error(f"Error checking PyTorch installation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    if verify_torch():
        print_success("PyTorch verification completed successfully")
    else:
        print_error("PyTorch verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main()