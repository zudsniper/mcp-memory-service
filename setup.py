#!/usr/bin/env python3
"""
Setup script for MCP Memory Service with cross-platform compatibility.
This script detects the system architecture and installs the appropriate dependencies.
"""
import os
import sys
import platform
import subprocess
from setuptools import setup, find_packages

# Determine the system architecture and platform
SYSTEM = platform.system().lower()
MACHINE = platform.machine().lower()
IS_WINDOWS = SYSTEM == "windows"
IS_MACOS = SYSTEM == "darwin"
IS_LINUX = SYSTEM == "linux"
IS_ARM = MACHINE in ("arm64", "aarch64")
IS_X86 = MACHINE in ("x86_64", "amd64", "x64")
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

# Check for CUDA availability
def has_cuda():
    """Check if CUDA is available on the system."""
    if IS_WINDOWS:
        return os.path.exists(os.environ.get('CUDA_PATH', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA'))
    elif IS_LINUX:
        return os.path.exists('/usr/local/cuda') or 'CUDA_HOME' in os.environ
    return False

# Check for ROCm availability
def has_rocm():
    """Check if ROCm is available on the system."""
    if not IS_LINUX:
        return False
    return os.path.exists('/opt/rocm') or 'ROCM_HOME' in os.environ

# Check for MPS availability (Apple Silicon)
def has_mps():
    """Check if MPS (Metal Performance Shaders) is available on Apple Silicon."""
    if not (IS_MACOS and IS_ARM):
        return False
    try:
        # Check if Metal is supported
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'],
            capture_output=True,
            text=True
        )
        return 'Metal' in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Determine PyTorch package based on platform
def get_torch_packages():
    """Get the appropriate PyTorch packages for the current platform."""
    packages = []
    
    # For Apple Silicon with MPS
    if IS_MACOS and IS_ARM and has_mps():
        print("Detected Apple Silicon with MPS support")
        packages.extend([
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0"
        ])
    # For macOS Intel
    elif IS_MACOS and IS_X86:
        print("Detected macOS with Intel CPU")
        packages.extend([
            "torch==2.0.1",  # Fixed version for Intel Macs
            "torchvision==0.15.2",
            "torchaudio==2.0.2",
            "sentence-transformers==2.2.2"  # Compatible version with torch 2.0.1
        ])
    # For CUDA-enabled systems
    elif has_cuda():
        print("Detected CUDA support")
        # Note: Windows with CUDA will be handled separately in install_requires
        if not IS_WINDOWS:
            # Linux with CUDA
            packages.extend([
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "torchaudio>=2.0.0"
            ])
    # For ROCm-enabled systems (Linux only)
    elif IS_LINUX and has_rocm():
        print("Detected ROCm support")
        packages.extend([
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0"
        ])
    # For Windows with DirectML
    elif IS_WINDOWS:
        try:
            # Check if DirectML is available
            import pkg_resources
            pkg_resources.get_distribution('torch-directml')
            print("Detected DirectML support")
            # Note: Base PyTorch for Windows will be handled separately
            packages.append("torch-directml>=0.2.0")
        except (ImportError, pkg_resources.DistributionNotFound):
            # Windows platforms need special handling
            print("Windows platform detected - PyTorch will be installed separately")
            # Don't add PyTorch packages here, they'll be installed separately
    # Default to CPU-only for all other platforms
    else:
        print("Using CPU-only PyTorch")
        packages.extend([
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0"
        ])
    
    return packages

# Get additional platform-specific packages
def get_platform_specific_packages():
    """Get additional platform-specific packages."""
    packages = []
    
    # For systems with limited resources, use ONNX Runtime
    if os.environ.get('MCP_MEMORY_USE_ONNX', '').lower() in ('1', 'true', 'yes'):
        packages.append("onnxruntime>=1.15.0")
    
    # For Windows with DirectML but without torch-directml already installed
    if IS_WINDOWS and not any('torch-directml' in pkg for pkg in packages):
        try:
            import pkg_resources
            pkg_resources.get_distribution('torch-directml')
        except (ImportError, pkg_resources.DistributionNotFound):
            if os.environ.get('MCP_MEMORY_USE_DIRECTML', '').lower() in ('1', 'true', 'yes'):
                packages.append("torch-directml>=0.2.0")
    
    return packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('torch')]

# Add platform-specific PyTorch packages
requirements.extend(get_torch_packages())

# Add other platform-specific packages
requirements.extend(get_platform_specific_packages())

# Print detected environment and packages
print(f"System: {SYSTEM} {MACHINE}")
print(f"Python: {PYTHON_VERSION}")
print("Installing packages:")
for req in requirements:
    print(f"  - {req}")

setup(
    name="mcp-memory-service",
    version="0.1.0",
    description="A semantic memory service using ChromaDB and sentence-transformers",
    author="Heinrich Krupp",
    author_email="heinrich.krupp@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "memory=mcp_memory_service.server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)