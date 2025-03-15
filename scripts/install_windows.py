#!/usr/bin/env python3
"""
Windows-specific installation script for MCP Memory Service.
This script handles the installation of PyTorch and MCP on Windows platforms.
"""
import os
import sys
import platform
import subprocess
import argparse

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def print_step(step, text):
    """Print a formatted step."""
    print(f"\n[{step}] {text}")

def print_info(text):
    """Print formatted info text."""
    print(f"  → {text}")

def print_error(text):
    """Print formatted error text."""
    print(f"  ❌ ERROR: {text}")

def print_success(text):
    """Print formatted success text."""
    print(f"  ✅ {text}")

def print_warning(text):
    """Print formatted warning text."""
    print(f"  ⚠️  {text}")

def check_system():
    """Check if running on Windows."""
    system = platform.system().lower()
    if system != "windows":
        print_error(f"This script is designed for Windows, but you're running on {system.capitalize()}")
        return False
    
    print_info(f"Running on {platform.system()} {platform.release()}")
    print_info(f"Python version: {platform.python_version()}")
    print_info(f"Architecture: {platform.machine()}")
    
    return True

def detect_cuda():
    """Detect CUDA availability and version."""
    cuda_available = False
    cuda_version = None
    
    # Check for CUDA_PATH environment variable
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path and os.path.exists(cuda_path):
        cuda_available = True
        try:
            # Try to get CUDA version
            nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc.exe')
            if os.path.exists(nvcc_path):
                nvcc_output = subprocess.check_output([nvcc_path, '--version'], 
                                                    stderr=subprocess.STDOUT, 
                                                    universal_newlines=True)
                for line in nvcc_output.split('\n'):
                    if 'release' in line:
                        cuda_version = line.split('release')[-1].strip().split(',')[0].strip()
                        break
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    # Try to detect CUDA using nvidia-smi
    if not cuda_available:
        try:
            nvidia_smi_output = subprocess.check_output(['nvidia-smi'], 
                                                      stderr=subprocess.DEVNULL, 
                                                      universal_newlines=True)
            cuda_available = True
            # Try to extract CUDA version from nvidia-smi output
            for line in nvidia_smi_output.split('\n'):
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[-1].strip()
                    break
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    if cuda_available:
        print_info(f"CUDA detected: {cuda_version or 'Unknown version'}")
    else:
        print_info("CUDA not detected, will use CPU-only mode")
    
    return {
        "available": cuda_available,
        "version": cuda_version
    }

def install_pytorch(cuda_info):
    """Install PyTorch with the appropriate index URL."""
    print_step("1", "Installing PyTorch for Windows")
    
    # Check if PyTorch is already installed
    try:
        import torch
        print_info(f"PyTorch is already installed (version {torch.__version__})")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print_success(f"CUDA is available (version {torch.version.cuda})")
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
            return True
        elif cuda_info["available"]:
            print_warning("CUDA is detected on the system but not available in PyTorch")
            print_info("Will attempt to reinstall PyTorch with CUDA support")
        else:
            print_info("Using CPU-only PyTorch")
            return True
    except ImportError:
        print_info("PyTorch is not installed, will install it now")
    
    # Determine the appropriate PyTorch index URL based on GPU
    cuda_channels = []
    if cuda_info["available"]:
        # Get CUDA version and determine appropriate index URL
        cuda_version = cuda_info.get("version", "")
        
        # Extract major version from CUDA version string
        cuda_major = None
        if cuda_version:
            # Try to extract the major version (e.g., "11.8" -> "11")
            try:
                cuda_major = cuda_version.split('.')[0]
            except (IndexError, AttributeError):
                pass
        
        # Add channels in order of preference
        if cuda_major == "12":
            cuda_channels = ["cu121", "cu118"]  # Try CUDA 12.x first, then fall back to 11.x
            print_info(f"Detected CUDA {cuda_version}, will try cu121 and cu118 channels")
        elif cuda_major == "11":
            cuda_channels = ["cu118", "cu117", "cu116"]  # Try different CUDA 11.x versions
            print_info(f"Detected CUDA {cuda_version}, will try cu118, cu117, and cu116 channels")
        elif cuda_major == "10":
            cuda_channels = ["cu102", "cu101"]  # Try different CUDA 10.x versions
            print_info(f"Detected CUDA {cuda_version}, will try cu102 and cu101 channels")
        else:
            # Default channels to try
            cuda_channels = ["cu118", "cu117", "cu121"]
            print_info(f"Using default channels for CUDA {cuda_version}")
    else:
        # CPU-only version
        cuda_channels = ["cpu"]
        print_info("Using CPU-only PyTorch for Windows")
    
    # Try different PyTorch versions
    torch_versions = ["2.1.0", "2.0.0", "1.13.1"]
    
    # Try different combinations of versions and channels
    for torch_version in torch_versions:
        for cuda_suffix in cuda_channels:
            index_url = f"https://download.pytorch.org/whl/{cuda_suffix}"
            
            try:
                print_info(f"Trying PyTorch {torch_version} with {cuda_suffix} channel")
                
                cmd = [
                    sys.executable, '-m', 'pip', 'install',
                    f"torch=={torch_version}",
                    f"torchvision=={torch_version}",
                    f"torchaudio=={torch_version}",
                    f"--index-url={index_url}"
                ]
                
                print_info(f"Running: {' '.join(cmd)}")
                subprocess.check_call(cmd)
                
                # Verify installation
                try:
                    import torch
                    print_success(f"PyTorch {torch.__version__} installed successfully")
                    
                    # Check if CUDA is available
                    if torch.cuda.is_available():
                        print_success(f"CUDA is available (version {torch.version.cuda})")
                        print_info(f"GPU: {torch.cuda.get_device_name(0)}")
                    elif cuda_suffix != "cpu":
                        print_warning(f"CUDA is not available in PyTorch despite using {cuda_suffix} channel")
                        continue  # Try next channel
                    
                    # Check if DirectML is needed for AMD/Intel GPUs
                    if cuda_suffix == "cpu" and not torch.cuda.is_available():
                        try:
                            # Check if we have an AMD or Intel GPU that could benefit from DirectML
                            dxdiag_output = subprocess.check_output(['dxdiag', '/t'],
                                                                stderr=subprocess.DEVNULL,
                                                                universal_newlines=True)
                            
                            # Check for AMD or Intel GPUs
                            if 'AMD' in dxdiag_output or 'Radeon' in dxdiag_output or 'Intel' in dxdiag_output:
                                print_info("AMD or Intel GPU detected, installing torch-directml")
                                subprocess.check_call([
                                    sys.executable, '-m', 'pip', 'install', 'torch-directml>=0.2.0'
                                ])
                        except (subprocess.SubprocessError, FileNotFoundError):
                            pass
                    
                    return True
                except ImportError:
                    print_error("PyTorch installation verification failed")
                    continue  # Try next version/channel
                
            except subprocess.SubprocessError as e:
                print_warning(f"Failed to install PyTorch {torch_version} with {cuda_suffix} channel: {e}")
                continue  # Try next version/channel
    
    # If we get here, all installation attempts failed
    print_error("All PyTorch installation attempts failed")
    print_warning("You may need to manually install PyTorch using instructions from https://pytorch.org/get-started/locally/")
    
    # Try to use any existing PyTorch installation
    try:
        import torch
        print_info(f"Using existing PyTorch installation (version {torch.__version__})")
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install other dependencies without PyTorch."""
    print_step("2", "Installing other dependencies")
    
    try:
        # Install requirements.txt but skip PyTorch
        print_info("Installing requirements.txt (excluding PyTorch)")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', 
            '--no-deps'  # Don't install dependencies to avoid PyTorch
        ])
        
        # Install MCP separately
        print_info("Installing MCP package")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 'mcp>=1.0.0,<2.0.0'
        ])
        
        print_success("Dependencies installed successfully")
        return True
    except subprocess.SubprocessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def install_package(dev_mode=False):
    """Install the MCP Memory Service package."""
    print_step("3", "Installing MCP Memory Service")
    
    try:
        # Install the package
        cmd = [sys.executable, '-m', 'pip', 'install']
        if dev_mode:
            cmd.append('-e')
        cmd.append('.')
        
        print_info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        print_success("MCP Memory Service installed successfully")
        return True
    except subprocess.SubprocessError as e:
        print_error(f"Failed to install MCP Memory Service: {e}")
        return False

def verify_installation():
    """Verify the installation."""
    print_step("4", "Verifying installation")
    
    # Run the verification script
    try:
        subprocess.check_call([
            sys.executable, 'scripts/verify_pytorch_windows.py'
        ])
        return True
    except subprocess.SubprocessError:
        print_error("Verification failed")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Windows-specific installation for MCP Memory Service")
    parser.add_argument('--dev', action='store_true', help='Install in development mode')
    args = parser.parse_args()
    
    print_header("MCP Memory Service Windows Installation")
    
    # Check if running on Windows
    if not check_system():
        sys.exit(1)
    
    # Detect CUDA
    cuda_info = detect_cuda()
    
    # Install PyTorch
    if not install_pytorch(cuda_info):
        print_warning("PyTorch installation failed, but will continue with other dependencies")
    
    # Install other dependencies
    if not install_dependencies():
        print_warning("Dependency installation failed, but will continue with package installation")
    
    # Install the package
    if not install_package(args.dev):
        print_error("Package installation failed")
        sys.exit(1)
    
    # Verify the installation
    if not verify_installation():
        print_warning("Installation verification failed, but installation may still work")
    
    print_header("Installation Complete")
    print_info("You can now run the MCP Memory Service using the 'memory' command")
    print_info("For more information, see the README.md file")

if __name__ == "__main__":
    main()