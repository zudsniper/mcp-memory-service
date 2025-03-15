#!/usr/bin/env python3
"""
Installation script for MCP Memory Service with cross-platform compatibility.
This script guides users through the installation process with the appropriate
dependencies for their platform.
"""
import os
import sys
import platform
import subprocess
import argparse
import shutil
from pathlib import Path

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

def detect_system():
    """Detect the system architecture and platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    is_windows = system == "windows"
    is_macos = system == "darwin"
    is_linux = system == "linux"
    is_arm = machine in ("arm64", "aarch64")
    is_x86 = machine in ("x86_64", "amd64", "x64")
    
    print_info(f"System: {platform.system()} {platform.release()}")
    print_info(f"Architecture: {machine}")
    print_info(f"Python: {python_version}")
    
    # Check for virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print_warning("Not running in a virtual environment. It's recommended to install in a virtual environment.")
    else:
        print_info(f"Virtual environment: {sys.prefix}")
    
    return {
        "system": system,
        "machine": machine,
        "python_version": python_version,
        "is_windows": is_windows,
        "is_macos": is_macos,
        "is_linux": is_linux,
        "is_arm": is_arm,
        "is_x86": is_x86,
        "in_venv": in_venv
    }

def detect_gpu():
    """Detect GPU and acceleration capabilities."""
    system_info = detect_system()
    
    # Check for CUDA
    has_cuda = False
    cuda_version = None
    if system_info["is_windows"]:
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path and os.path.exists(cuda_path):
            has_cuda = True
            try:
                # Try to get CUDA version
                nvcc_output = subprocess.check_output([os.path.join(cuda_path, 'bin', 'nvcc'), '--version'], 
                                                     stderr=subprocess.STDOUT, 
                                                     universal_newlines=True)
                for line in nvcc_output.split('\n'):
                    if 'release' in line:
                        cuda_version = line.split('release')[-1].strip().split(',')[0].strip()
                        break
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
    elif system_info["is_linux"]:
        cuda_paths = ['/usr/local/cuda', os.environ.get('CUDA_HOME')]
        for path in cuda_paths:
            if path and os.path.exists(path):
                has_cuda = True
                try:
                    # Try to get CUDA version
                    nvcc_output = subprocess.check_output([os.path.join(path, 'bin', 'nvcc'), '--version'], 
                                                         stderr=subprocess.STDOUT, 
                                                         universal_newlines=True)
                    for line in nvcc_output.split('\n'):
                        if 'release' in line:
                            cuda_version = line.split('release')[-1].strip().split(',')[0].strip()
                            break
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
                break
    
    # Check for ROCm (AMD)
    has_rocm = False
    rocm_version = None
    if system_info["is_linux"]:
        rocm_paths = ['/opt/rocm', os.environ.get('ROCM_HOME')]
        for path in rocm_paths:
            if path and os.path.exists(path):
                has_rocm = True
                try:
                    # Try to get ROCm version
                    with open(os.path.join(path, 'bin', '.rocmversion'), 'r') as f:
                        rocm_version = f.read().strip()
                except (FileNotFoundError, IOError):
                    try:
                        rocm_output = subprocess.check_output(['rocminfo'], 
                                                            stderr=subprocess.STDOUT, 
                                                            universal_newlines=True)
                        for line in rocm_output.split('\n'):
                            if 'Version' in line:
                                rocm_version = line.split(':')[-1].strip()
                                break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        pass
                break
    
    # Check for MPS (Apple Silicon)
    has_mps = False
    if system_info["is_macos"] and system_info["is_arm"]:
        try:
            # Check if Metal is supported
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True,
                text=True
            )
            has_mps = 'Metal' in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    # Check for DirectML (Windows)
    has_directml = False
    if system_info["is_windows"]:
        try:
            # Check if DirectML package is installed
            import pkg_resources
            pkg_resources.get_distribution('torch-directml')
            has_directml = True
        except (ImportError, pkg_resources.DistributionNotFound):
            # Check if DirectML is available on the system
            try:
                import ctypes
                ctypes.WinDLL('DirectML.dll')
                has_directml = True
            except (ImportError, OSError):
                pass
    
    # Print GPU information
    if has_cuda:
        print_info(f"CUDA detected: {cuda_version or 'Unknown version'}")
    if has_rocm:
        print_info(f"ROCm detected: {rocm_version or 'Unknown version'}")
    if has_mps:
        print_info("Apple Metal Performance Shaders (MPS) detected")
    if has_directml:
        print_info("DirectML detected")
    
    if not (has_cuda or has_rocm or has_mps or has_directml):
        print_info("No GPU acceleration detected, will use CPU-only mode")
    
    return {
        "has_cuda": has_cuda,
        "cuda_version": cuda_version,
        "has_rocm": has_rocm,
        "rocm_version": rocm_version,
        "has_mps": has_mps,
        "has_directml": has_directml
    }

def check_dependencies():
    """Check for required dependencies."""
    print_step("2", "Checking dependencies")
    
    # Check for pip
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        print_info("pip is installed")
    except subprocess.SubprocessError:
        print_error("pip is not installed. Please install pip first.")
        return False
    
    # Check for setuptools
    try:
        import setuptools
        print_info(f"setuptools is installed: {setuptools.__version__}")
    except ImportError:
        print_warning("setuptools is not installed. Will attempt to install it.")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'setuptools'], 
                                 stdout=subprocess.DEVNULL)
            print_success("setuptools installed successfully")
        except subprocess.SubprocessError:
            print_error("Failed to install setuptools. Please install it manually.")
            return False
    
    # Check for wheel
    try:
        import wheel
        print_info(f"wheel is installed: {wheel.__version__}")
    except ImportError:
        print_warning("wheel is not installed. Will attempt to install it.")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wheel'], 
                                 stdout=subprocess.DEVNULL)
            print_success("wheel installed successfully")
        except subprocess.SubprocessError:
            print_error("Failed to install wheel. Please install it manually.")
            return False
    
    return True

def install_pytorch_windows(gpu_info):
    """Install PyTorch on Windows using the appropriate index URL."""
    print_step("3a", "Installing PyTorch for Windows")
    
    # Determine the appropriate PyTorch index URL based on GPU
    if gpu_info["has_cuda"]:
        # Get CUDA version and determine appropriate index URL
        cuda_version = gpu_info.get("cuda_version", "")
        
        # Extract major version from CUDA version string
        cuda_major = None
        if cuda_version:
            # Try to extract the major version (e.g., "11.8" -> "11")
            try:
                cuda_major = cuda_version.split('.')[0]
            except (IndexError, AttributeError):
                pass
        
        # Default to cu118 if we couldn't determine the version or it's not a common one
        if cuda_major == "12":
            cuda_suffix = "cu121"  # CUDA 12.x
            print_info(f"Detected CUDA {cuda_version}, using cu121 channel")
        elif cuda_major == "11":
            cuda_suffix = "cu118"  # CUDA 11.x
            print_info(f"Detected CUDA {cuda_version}, using cu118 channel")
        elif cuda_major == "10":
            cuda_suffix = "cu102"  # CUDA 10.x
            print_info(f"Detected CUDA {cuda_version}, using cu102 channel")
        else:
            # Default to cu118 as a safe choice for newer NVIDIA GPUs
            cuda_suffix = "cu118"
            print_info(f"Using default cu118 channel for CUDA {cuda_version}")
            
        index_url = f"https://download.pytorch.org/whl/{cuda_suffix}"
    else:
        # CPU-only version
        index_url = "https://download.pytorch.org/whl/cpu"
        print_info("Using CPU-only PyTorch for Windows")
    
    # Install PyTorch with the appropriate index URL
    try:
        # Use a stable version that's known to have Windows wheels
        torch_version = "2.1.0"  # This version has Windows wheels available
        
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            f"torch=={torch_version}",
            f"torchvision=={torch_version}",
            f"torchaudio=={torch_version}",
            f"--index-url={index_url}"
        ]
        
        print_info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        # Check if DirectML is needed
        if gpu_info["has_directml"]:
            print_info("Installing torch-directml for DirectML support")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 'torch-directml>=0.2.0'
            ])
            
        print_success("PyTorch installed successfully for Windows")
        return True
    except subprocess.SubprocessError as e:
        print_error(f"Failed to install PyTorch for Windows: {e}")
        print_warning("You may need to manually install PyTorch using instructions from https://pytorch.org/get-started/locally/")
        return False

def install_package(args):
    """Install the package with the appropriate dependencies."""
    print_step("3", "Installing MCP Memory Service")
    
    # Determine installation mode
    install_mode = []
    if args.dev:
        install_mode = ['-e']
        print_info("Installing in development mode")
    
    # Set environment variables for installation
    env = os.environ.copy()
    
    # Set environment variables based on detected GPU
    gpu_info = detect_gpu()
    if gpu_info["has_cuda"]:
        print_info("Configuring for CUDA installation")
    elif gpu_info["has_rocm"]:
        print_info("Configuring for ROCm installation")
        env['MCP_MEMORY_USE_ROCM'] = '1'
    elif gpu_info["has_mps"]:
        print_info("Configuring for Apple Silicon MPS installation")
        env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    elif gpu_info["has_directml"]:
        print_info("Configuring for DirectML installation")
        env['MCP_MEMORY_USE_DIRECTML'] = '1'
    else:
        print_info("Configuring for CPU-only installation")
        env['MCP_MEMORY_USE_ONNX'] = '1'
    
    # For Windows, install PyTorch separately with the appropriate index URL
    system_info = detect_system()
    if system_info["is_windows"]:
        if not install_pytorch_windows(gpu_info):
            print_warning("PyTorch installation for Windows failed, but will continue with package installation")
    
    # Install the package
    try:
        cmd = [sys.executable, '-m', 'pip', 'install'] + install_mode + ['.']
        print_info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd, env=env)
        print_success("MCP Memory Service installed successfully")
        return True
    except subprocess.SubprocessError as e:
        print_error(f"Failed to install MCP Memory Service: {e}")
        
        # Provide more helpful error message for Windows users
        if system_info["is_windows"] and "torch" in str(e):
            print_warning("The error appears to be related to PyTorch installation on Windows.")
            print_info("You can try manually installing PyTorch first using:")
            print_info("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print_info("Then run this installation script again.")
        
        return False

def configure_paths(args):
    """Configure paths for the MCP Memory Service."""
    print_step("4", "Configuring paths")
    
    # Determine home directory
    home_dir = Path.home()
    
    # Determine base directory based on platform
    if platform.system() == 'Darwin':  # macOS
        base_dir = home_dir / 'Library' / 'Application Support' / 'mcp-memory'
    elif platform.system() == 'Windows':  # Windows
        base_dir = Path(os.environ.get('LOCALAPPDATA', '')) / 'mcp-memory'
    else:  # Linux and others
        base_dir = home_dir / '.local' / 'share' / 'mcp-memory'
    
    # Create directories
    chroma_path = args.chroma_path or (base_dir / 'chroma_db')
    backups_path = args.backups_path or (base_dir / 'backups')
    
    try:
        os.makedirs(chroma_path, exist_ok=True)
        os.makedirs(backups_path, exist_ok=True)
        print_info(f"ChromaDB path: {chroma_path}")
        print_info(f"Backups path: {backups_path}")
        
        # Test if directories are writable
        test_file = os.path.join(chroma_path, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        test_file = os.path.join(backups_path, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        print_success("Directories created and are writable")
        
        # Configure Claude Desktop if available
        claude_config_paths = [
            home_dir / 'Library' / 'Application Support' / 'Claude' / 'claude_desktop_config.json',
            home_dir / '.config' / 'Claude' / 'claude_desktop_config.json',
            Path('claude_config') / 'claude_desktop_config.json'
        ]
        
        for config_path in claude_config_paths:
            if config_path.exists():
                print_info(f"Found Claude Desktop config at {config_path}")
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Update or add MCP Memory configuration
                    if 'mcpServers' not in config:
                        config['mcpServers'] = {}
                    
                    # Create or update the memory server configuration
                    if system_info["is_windows"]:
                        # Use the memory_wrapper.py script for Windows
                        script_path = os.path.abspath("memory_wrapper.py")
                        config['mcpServers']['memory'] = {
                            "command": "python",
                            "args": [script_path],
                            "env": {
                                "MCP_MEMORY_CHROMA_PATH": str(chroma_path),
                                "MCP_MEMORY_BACKUPS_PATH": str(backups_path)
                            }
                        }
                        print_info("Configured Claude Desktop to use memory_wrapper.py for Windows")
                    else:
                        # Use the standard configuration for other platforms
                        config['mcpServers']['memory'] = {
                            "command": "uv",
                            "args": [
                                "--directory",
                                os.path.abspath("."),
                                "run",
                                "memory"
                            ],
                            "env": {
                                "MCP_MEMORY_CHROMA_PATH": str(chroma_path),
                                "MCP_MEMORY_BACKUPS_PATH": str(backups_path)
                            }
                        }
                    
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    print_success("Updated Claude Desktop configuration")
                except Exception as e:
                    print_warning(f"Failed to update Claude Desktop configuration: {e}")
                break
        
        return True
    except Exception as e:
        print_error(f"Failed to configure paths: {e}")
        return False

def verify_installation():
    """Verify the installation."""
    print_step("5", "Verifying installation")
    
    # Check if the package is installed
    try:
        import mcp_memory_service
        print_success(f"MCP Memory Service is installed: {mcp_memory_service.__file__}")
    except ImportError:
        print_error("MCP Memory Service is not installed correctly")
        return False
    
    # Check if the entry point is available
    memory_script = shutil.which('memory')
    if memory_script:
        print_success(f"Memory command is available: {memory_script}")
    else:
        print_warning("Memory command is not available in PATH")
    
    # Check if PyTorch is installed correctly
    try:
        import torch
        print_info(f"PyTorch is installed: {torch.__version__}")
        
        # Check for CUDA
        if torch.cuda.is_available():
            print_success(f"CUDA is available: {torch.version.cuda}")
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_success("MPS (Metal Performance Shaders) is available")
        # Check for DirectML
        else:
            try:
                import torch_directml
                print_success(f"DirectML is available: {torch_directml.__version__}")
            except ImportError:
                print_info("Using CPU-only PyTorch")
    except ImportError:
        print_error("PyTorch is not installed correctly")
        return False
    
    # Check if sentence-transformers is installed correctly
    try:
        import sentence_transformers
        print_success(f"sentence-transformers is installed: {sentence_transformers.__version__}")
    except ImportError:
        print_error("sentence-transformers is not installed correctly")
        return False
    
    # Check if ChromaDB is installed correctly
    try:
        import chromadb
        print_success(f"ChromaDB is installed: {chromadb.__version__}")
    except ImportError:
        print_error("ChromaDB is not installed correctly")
        return False
    
    print_success("Installation verification completed successfully")
    return True

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Install MCP Memory Service")
    parser.add_argument('--dev', action='store_true', help='Install in development mode')
    parser.add_argument('--chroma-path', type=str, help='Path to ChromaDB storage')
    parser.add_argument('--backups-path', type=str, help='Path to backups storage')
    args = parser.parse_args()
    
    print_header("MCP Memory Service Installation")
    
    # Step 1: Detect system
    print_step("1", "Detecting system")
    system_info = detect_system()
    
    # Step 2: Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 3: Install package
    if not install_package(args):
        sys.exit(1)
    
    # Step 4: Configure paths
    if not configure_paths(args):
        print_warning("Path configuration failed, but installation may still work")
    
    # Step 5: Verify installation
    if not verify_installation():
        print_warning("Installation verification failed, but installation may still work")
    
    print_header("Installation Complete")
    print_info("You can now run the MCP Memory Service using the 'memory' command")
    print_info("For more information, see the README.md file")

if __name__ == "__main__":
    main()