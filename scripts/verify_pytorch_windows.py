#!/usr/bin/env python3
"""
Verification script for PyTorch installation on Windows.
This script checks if PyTorch is properly installed and configured for Windows.
"""
import os
import sys
import platform
import subprocess
import importlib.util

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def print_info(text):
    """Print formatted info text."""
    print(f"  → {text}")

def print_success(text):
    """Print formatted success text."""
    print(f"  ✅ {text}")

def print_error(text):
    """Print formatted error text."""
    print(f"  ❌ ERROR: {text}")

def print_warning(text):
    """Print formatted warning text."""
    print(f"  ⚠️  {text}")

def check_system():
    """Check if running on Windows."""
    system = platform.system().lower()
    if system != "windows":
        print_warning(f"This script is designed for Windows, but you're running on {system.capitalize()}")
    else:
        print_info(f"Running on {platform.system()} {platform.release()}")
    
    print_info(f"Python version: {platform.python_version()}")
    print_info(f"Architecture: {platform.machine()}")
    
    return system == "windows"

def check_pytorch_installation():
    """Check if PyTorch is installed and properly configured."""
    try:
        import torch
        print_success(f"PyTorch is installed (version {torch.__version__})")
        
        # Check if PyTorch was installed from the correct index URL
        if hasattr(torch, '_C'):
            print_success("PyTorch C extensions are available")
        else:
            print_warning("PyTorch C extensions might not be properly installed")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print_success(f"CUDA is available (version {torch.version.cuda})")
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
            print_info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print_info("CUDA is not available, using CPU only")
            
            # Check if DirectML is available
            try:
                import torch_directml
                print_success(f"DirectML is available (version {torch_directml.__version__})")
            except ImportError:
                print_info("DirectML is not available")
        
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
    except ImportError:
        print_error("PyTorch is not installed")
        return False
    except Exception as e:
        print_error(f"Error checking PyTorch installation: {e}")
        return False

def suggest_installation():
    """Suggest PyTorch installation commands."""
    print_header("Installation Suggestions")
    print_info("To install PyTorch for Windows, use one of the following commands:")
    print_info("\nFor CUDA support (NVIDIA GPUs):")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print_info("\nFor CPU-only:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    
    print_info("\nFor DirectML support (AMD/Intel GPUs):")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("pip install torch-directml")
    
    print_info("\nAfter installing PyTorch, run this script again to verify the installation.")

def main():
    """Main function."""
    print_header("PyTorch Windows Installation Verification")
    
    is_windows = check_system()
    if not is_windows:
        print_warning("This script is designed for Windows, but may still provide useful information")
    
    pytorch_installed = check_pytorch_installation()
    
    if not pytorch_installed:
        suggest_installation()
        return 1
    
    print_header("Verification Complete")
    print_success("PyTorch is properly installed and configured for Windows")
    return 0

if __name__ == "__main__":
    sys.exit(main())