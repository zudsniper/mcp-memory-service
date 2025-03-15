#!/usr/bin/env python3
"""
Debug script to identify PyTorch dependency issues.
This script analyzes the dependency chain to find what's requiring PyTorch 2.5.1.
"""
import sys
import os
import importlib
import pkg_resources
import subprocess
import traceback
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

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

def check_installed_pytorch():
    """Check if PyTorch is installed and get its version."""
    try:
        import torch
        print_success(f"PyTorch is installed (version {torch.__version__})")
        print_info(f"PyTorch location: {torch.__file__}")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print_success(f"CUDA is available (version {torch.version.cuda})")
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print_info("CUDA is not available, using CPU-only mode")
        
        return True, torch.__version__
    except ImportError:
        print_error("PyTorch is not installed")
        return False, None
    except Exception as e:
        print_error(f"Error checking PyTorch installation: {e}")
        return False, None

def analyze_dependencies():
    """Analyze dependencies to find what's requiring PyTorch 2.5.1."""
    print_header("Analyzing Dependencies")
    
    # Get all installed packages
    installed_packages = {pkg.key: pkg for pkg in pkg_resources.working_set}
    
    # Check for direct PyTorch dependencies
    torch_dependents = []
    for pkg_name, pkg in installed_packages.items():
        for req in pkg.requires():
            if req.name == 'torch':
                torch_dependents.append((pkg_name, str(req)))
    
    if torch_dependents:
        print_info(f"Found {len(torch_dependents)} packages that directly depend on PyTorch:")
        for pkg_name, req in torch_dependents:
            print_info(f"  - {pkg_name} requires {req}")
    else:
        print_info("No packages directly depend on PyTorch")
    
    # Check for specific version requirements
    problematic_deps = []
    for pkg_name, req in torch_dependents:
        if "2.5.1" in req:
            problematic_deps.append((pkg_name, req))
    
    if problematic_deps:
        print_warning(f"Found {len(problematic_deps)} packages that specifically require PyTorch 2.5.1:")
        for pkg_name, req in problematic_deps:
            print_warning(f"  - {pkg_name} requires {req}")
    else:
        print_success("No packages specifically require PyTorch 2.5.1")
    
    # Check for sentence-transformers dependencies
    try:
        import sentence_transformers
        print_info(f"sentence-transformers version: {sentence_transformers.__version__}")
        print_info(f"sentence-transformers location: {sentence_transformers.__file__}")
        
        # Check transformers version
        try:
            import transformers
            print_info(f"transformers version: {transformers.__version__}")
            print_info(f"transformers location: {transformers.__file__}")
        except ImportError:
            print_error("transformers is not installed")
    except ImportError:
        print_error("sentence-transformers is not installed")
    
    # Check for mcp-memory-service dependencies
    try:
        import mcp_memory_service
        print_info(f"mcp-memory-service location: {mcp_memory_service.__file__}")
    except ImportError:
        print_error("mcp-memory-service is not installed")
    
    # Check for chromadb dependencies
    try:
        import chromadb
        print_info(f"chromadb version: {chromadb.__version__}")
        print_info(f"chromadb location: {chromadb.__file__}")
    except ImportError:
        print_error("chromadb is not installed")

def check_pip_config():
    """Check pip configuration for any issues."""
    print_header("Checking pip Configuration")
    
    try:
        # Run pip config list to see current configuration
        result = subprocess.run([sys.executable, '-m', 'pip', 'config', 'list'], 
                               capture_output=True, text=True)
        
        if result.stdout.strip():
            print_info("Pip configuration:")
            for line in result.stdout.strip().split('\n'):
                print_info(f"  {line}")
        else:
            print_info("No custom pip configuration found")
        
        # Check for index-url settings
        if "index-url" in result.stdout:
            print_warning("Custom index-url found in pip configuration")
        
        # Check for extra-index-url settings
        if "extra-index-url" in result.stdout:
            print_warning("Custom extra-index-url found in pip configuration")
    except subprocess.SubprocessError as e:
        print_error(f"Error checking pip configuration: {e}")

def check_import_process():
    """Check what happens during the import process."""
    print_header("Checking Import Process")
    
    # Create a temporary script to import torch and print debug info
    temp_script = """
import sys
import os
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"sys.path:")
for p in sys.path:
    print(f"  - {p}")
print("\\nAttempting to import torch...")
try:
    import torch
    print(f"Successfully imported torch {torch.__version__}")
    print(f"torch location: {torch.__file__}")
except Exception as e:
    print(f"Error importing torch: {e}")
    import traceback
    traceback.print_exc()
"""
    
    temp_file = Path("temp_import_test.py")
    try:
        # Write the temporary script
        with open(temp_file, "w") as f:
            f.write(temp_script)
        
        # Run the script with subprocess to see what happens
        print_info("Running import test script...")
        result = subprocess.run([sys.executable, str(temp_file)], 
                               capture_output=True, text=True)
        
        print_info("Import test output:")
        for line in result.stdout.strip().split('\n'):
            print_info(f"  {line}")
        
        if result.stderr:
            print_warning("Import test errors:")
            for line in result.stderr.strip().split('\n'):
                print_warning(f"  {line}")
    except Exception as e:
        print_error(f"Error running import test: {e}")
    finally:
        # Clean up the temporary file
        if temp_file.exists():
            temp_file.unlink()

def check_mcp_server_import():
    """Check what happens when importing mcp_memory_service.server."""
    print_header("Checking MCP Server Import")
    
    # Create a temporary script to import mcp_memory_service.server and print debug info
    temp_script = """
import sys
import os
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"sys.path:")
for p in sys.path:
    print(f"  - {p}")
print("\\nAttempting to import mcp_memory_service.server...")
try:
    import mcp_memory_service.server
    print(f"Successfully imported mcp_memory_service.server")
    print(f"mcp_memory_service.server location: {mcp_memory_service.server.__file__}")
    
    # Check for torch import in server.py
    import inspect
    server_code = inspect.getsource(mcp_memory_service.server)
    if "import torch" in server_code:
        print("server.py directly imports torch")
    else:
        print("server.py does not directly import torch")
except Exception as e:
    print(f"Error importing mcp_memory_service.server: {e}")
    import traceback
    traceback.print_exc()
"""
    
    temp_file = Path("temp_server_import_test.py")
    try:
        # Write the temporary script
        with open(temp_file, "w") as f:
            f.write(temp_script)
        
        # Run the script with subprocess to see what happens
        print_info("Running server import test script...")
        result = subprocess.run([sys.executable, str(temp_file)], 
                               capture_output=True, text=True)
        
        print_info("Server import test output:")
        for line in result.stdout.strip().split('\n'):
            print_info(f"  {line}")
        
        if result.stderr:
            print_warning("Server import test errors:")
            for line in result.stderr.strip().split('\n'):
                print_warning(f"  {line}")
    except Exception as e:
        print_error(f"Error running server import test: {e}")
    finally:
        # Clean up the temporary file
        if temp_file.exists():
            temp_file.unlink()

def check_sentence_transformers_import():
    """Check what happens when importing sentence_transformers."""
    print_header("Checking sentence_transformers Import")
    
    # Create a temporary script to import sentence_transformers and print debug info
    temp_script = """
import sys
import os
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("\\nAttempting to import sentence_transformers...")
try:
    import sentence_transformers
    print(f"Successfully imported sentence_transformers {sentence_transformers.__version__}")
    print(f"sentence_transformers location: {sentence_transformers.__file__}")
    
    # Check dependencies
    import pkg_resources
    st_pkg = pkg_resources.get_distribution('sentence-transformers')
    print("\\nsentence-transformers dependencies:")
    for req in st_pkg.requires():
        print(f"  - {req}")
except Exception as e:
    print(f"Error importing sentence_transformers: {e}")
    import traceback
    traceback.print_exc()
"""
    
    temp_file = Path("temp_st_import_test.py")
    try:
        # Write the temporary script
        with open(temp_file, "w") as f:
            f.write(temp_script)
        
        # Run the script with subprocess to see what happens
        print_info("Running sentence_transformers import test script...")
        result = subprocess.run([sys.executable, str(temp_file)], 
                               capture_output=True, text=True)
        
        print_info("sentence_transformers import test output:")
        for line in result.stdout.strip().split('\n'):
            print_info(f"  {line}")
        
        if result.stderr:
            print_warning("sentence_transformers import test errors:")
            for line in result.stderr.strip().split('\n'):
                print_warning(f"  {line}")
    except Exception as e:
        print_error(f"Error running sentence_transformers import test: {e}")
    finally:
        # Clean up the temporary file
        if temp_file.exists():
            temp_file.unlink()

def main():
    """Main function."""
    print_header("PyTorch Dependency Debugger")
    
    # Check if PyTorch is installed
    pytorch_installed, pytorch_version = check_installed_pytorch()
    
    # Analyze dependencies
    analyze_dependencies()
    
    # Check pip configuration
    check_pip_config()
    
    # Check import process
    check_import_process()
    
    # Check mcp_memory_service.server import
    check_mcp_server_import()
    
    # Check sentence_transformers import
    check_sentence_transformers_import()
    
    print_header("Debug Complete")
    print_info("Check the output above for any issues with PyTorch dependencies")
    print_info("Look for packages that specifically require PyTorch 2.5.1")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_error(f"Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)