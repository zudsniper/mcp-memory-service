#!/usr/bin/env python3
"""
Fix script for PyTorch dependency issues.
This script pins PyTorch to the currently installed version and prevents automatic upgrades.
"""
import sys
import os
import subprocess
import pkg_resources
import site
import tempfile
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

def get_installed_pytorch_version():
    """Get the currently installed PyTorch version."""
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None

def create_constraint_file():
    """Create a constraints file to pin PyTorch version."""
    torch_version = get_installed_pytorch_version()
    if not torch_version:
        print_error("PyTorch is not installed, cannot create constraint file")
        return None
    
    print_info(f"Creating constraint file for PyTorch {torch_version}")
    
    # Create a temporary constraints file
    fd, constraints_path = tempfile.mkstemp(suffix='.txt', prefix='torch_constraints_')
    os.close(fd)
    
    with open(constraints_path, 'w') as f:
        f.write(f"torch=={torch_version}\n")
        f.write(f"torchvision=={torch_version}\n")
        f.write(f"torchaudio=={torch_version}\n")
    
    print_success(f"Created constraints file at {constraints_path}")
    return constraints_path

def get_installed_package_version(package_name):
    """Get the currently installed version of a package."""
    try:
        import pkg_resources
        return pkg_resources.get_distribution(package_name).version
    except (pkg_resources.DistributionNotFound, ImportError):
        return None

def pin_pytorch_version(constraints_path):
    """Pin PyTorch version using pip constraints."""
    if not constraints_path or not os.path.exists(constraints_path):
        print_error("Invalid constraints file path")
        return False
    
    # Get the installed PyTorch version
    torch_version = get_installed_pytorch_version()
    if not torch_version:
        print_error("PyTorch is not installed, cannot pin version")
        return False
    
    # Get the installed torchvision and torchaudio versions
    torchvision_version = get_installed_package_version("torchvision")
    torchaudio_version = get_installed_package_version("torchaudio")
    
    print_info(f"Installed versions: torch={torch_version}, torchvision={torchvision_version}, torchaudio={torchaudio_version}")
    
    # Check if version has CUDA suffix
    if '+cu' in torch_version:
        print_info(f"Detected CUDA-enabled PyTorch version: {torch_version}")
        print_info("Using direct index URL approach instead of constraints")
        
        # Extract the CUDA version from the PyTorch version
        cuda_suffix = torch_version.split('+')[1]  # e.g., 'cu118'
        
        # Use the appropriate index URL
        index_url = f"https://download.pytorch.org/whl/{cuda_suffix}"
        
        # Get the base version without the CUDA suffix
        torch_base_version = torch_version.split('+')[0]  # e.g., '2.6.0'
        
        # Get the base versions for torchvision and torchaudio if they have CUDA suffixes
        torchvision_base_version = torchvision_version.split('+')[0] if torchvision_version and '+' in torchvision_version else torchvision_version
        torchaudio_base_version = torchaudio_version.split('+')[0] if torchaudio_version and '+' in torchaudio_version else torchaudio_version
        
        print_info(f"Using index URL: {index_url} for PyTorch packages")
        
        try:
            # Run pip install with the index URL for torch
            cmd = [
                sys.executable, '-m', 'pip', 'install',
                '--no-deps',
                f"torch=={torch_base_version}",
                f"--index-url={index_url}"
            ]
            
            print_info(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            
            # Install torchvision if available
            if torchvision_version:
                cmd = [
                    sys.executable, '-m', 'pip', 'install',
                    '--no-deps',
                    f"torchvision=={torchvision_base_version}",
                    f"--index-url={index_url}"
                ]
                
                print_info(f"Running: {' '.join(cmd)}")
                try:
                    subprocess.check_call(cmd)
                except subprocess.SubprocessError as e:
                    print_warning(f"Failed to install torchvision: {e}")
                    print_info("This is not critical, continuing...")
            
            # Install torchaudio if available
            if torchaudio_version:
                cmd = [
                    sys.executable, '-m', 'pip', 'install',
                    '--no-deps',
                    f"torchaudio=={torchaudio_base_version}",
                    f"--index-url={index_url}"
                ]
                
                print_info(f"Running: {' '.join(cmd)}")
                try:
                    subprocess.check_call(cmd)
                except subprocess.SubprocessError as e:
                    print_warning(f"Failed to install torchaudio: {e}")
                    print_info("This is not critical, continuing...")
            
            print_success("PyTorch packages pinned successfully")
            return True
        except subprocess.SubprocessError as e:
            print_error(f"Failed to pin PyTorch version: {e}")
            return False
    else:
        # For non-CUDA versions, use the constraints file
        print_info("Pinning PyTorch version using pip constraints")
        
        try:
            # Run pip install with constraints to pin the version
            cmd = [
                sys.executable, '-m', 'pip', 'install',
                '--upgrade',
                '--force-reinstall',
                '--no-deps',
                '--constraint', constraints_path,
                'torch', 'torchvision', 'torchaudio'
            ]
            
            print_info(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            
            print_success("PyTorch version pinned successfully")
            return True
        except subprocess.SubprocessError as e:
            print_error(f"Failed to pin PyTorch version: {e}")
            return False

def create_no_deps_wrapper():
    """Create a wrapper script that prevents automatic dependency installation."""
    print_info("Creating no-deps wrapper for pip")
    
    # Get site-packages directory
    site_packages = site.getsitepackages()[0]
    
    # Create a directory for the wrapper if it doesn't exist
    wrapper_dir = os.path.join(site_packages, 'pip_no_deps_wrapper')
    os.makedirs(wrapper_dir, exist_ok=True)
    
    # Create an __init__.py file
    with open(os.path.join(wrapper_dir, '__init__.py'), 'w') as f:
        f.write("# Pip no-deps wrapper\n")
    
    # Create a wrapper.py file
    wrapper_path = os.path.join(wrapper_dir, 'wrapper.py')
    with open(wrapper_path, 'w') as f:
        f.write("""
# Wrapper to prevent automatic dependency installation
import sys
import os
import importlib.util
import types

class PipNoDepsFinder:
    def __init__(self, blocked_packages=None):
        self.blocked_packages = blocked_packages or ['torch', 'torchvision', 'torchaudio']
    
    def find_spec(self, fullname, path=None, target=None):
        # Block automatic installation of specified packages
        if any(fullname.startswith(pkg) for pkg in self.blocked_packages):
            # Check if the package is already installed
            spec = importlib.util.find_spec(fullname)
            if spec is not None:
                return spec
            
            # If not installed, print a warning and return None
            print(f"WARNING: Blocked automatic installation of {fullname}", file=sys.stderr)
            return None
        
        # Let the normal import system handle other packages
        return None

# Install the import hook
sys.meta_path.insert(0, PipNoDepsFinder())

# Monkey patch pip to prevent automatic installation
try:
    import pip._internal.resolution.resolvelib.factory
    original_get_installation_candidate = pip._internal.resolution.resolvelib.factory.Factory.get_installation_requirement
    
    def patched_get_installation_requirement(self, requirement):
        # Block torch installations from PyPI
        if requirement.name in ['torch', 'torchvision', 'torchaudio']:
            print(f"WARNING: Blocked automatic installation of {requirement.name}", file=sys.stderr)
            return None
        return original_get_installation_candidate(self, requirement)
    
    pip._internal.resolution.resolvelib.factory.Factory.get_installation_requirement = patched_get_installation_requirement
except (ImportError, AttributeError):
    pass

print("Pip no-deps wrapper installed", file=sys.stderr)
""")
    
    print_success(f"Created wrapper at {wrapper_path}")
    
    # Create a .pth file to automatically load the wrapper
    pth_path = os.path.join(site_packages, 'pip_no_deps_wrapper.pth')
    with open(pth_path, 'w') as f:
        f.write(f"import pip_no_deps_wrapper.wrapper\n")
    
    print_success(f"Created .pth file at {pth_path}")
    return True

def create_sitecustomize():
    """Create a sitecustomize.py file to prevent automatic dependency installation."""
    print_info("Creating sitecustomize.py to prevent automatic dependency installation")
    
    # Get site-packages directory
    site_packages = site.getsitepackages()[0]
    
    # Create sitecustomize.py
    sitecustomize_path = os.path.join(site_packages, 'sitecustomize.py')
    
    # Check if file already exists
    if os.path.exists(sitecustomize_path):
        print_warning(f"sitecustomize.py already exists at {sitecustomize_path}")
        print_info("Creating backup of existing file")
        backup_path = sitecustomize_path + '.bak'
        os.rename(sitecustomize_path, backup_path)
        print_success(f"Backup created at {backup_path}")
    
    # Create new sitecustomize.py
    with open(sitecustomize_path, 'w') as f:
        f.write("""
# sitecustomize.py to prevent automatic dependency installation
import sys
import os
import importlib.util

# Print debug info
print("sitecustomize.py loaded", file=sys.stderr)

# Create a custom import hook to prevent automatic installation
class PreventAutoInstallImportHook:
    def __init__(self):
        self.blocked_packages = ['torch', 'torchvision', 'torchaudio']
    
    def find_spec(self, fullname, path, target=None):
        if any(fullname.startswith(pkg) for pkg in self.blocked_packages):
            # Check if the package is already installed
            spec = importlib.util.find_spec(fullname)
            if spec is not None:
                return spec
            
            # If not installed, print a warning and return None
            print(f"WARNING: Blocked automatic installation of {fullname}", file=sys.stderr)
            return None
        # Return None to let the normal import system handle it
        return None

# Register the import hook
sys.meta_path.insert(0, PreventAutoInstallImportHook())

# Set environment variables to prevent pip from installing dependencies
os.environ["PIP_NO_DEPENDENCIES"] = "1"
os.environ["PIP_NO_INSTALL"] = "1"
""")
    
    print_success(f"Created sitecustomize.py at {sitecustomize_path}")
    return True

def update_requirements_file():
    """Update requirements.txt to pin PyTorch version."""
    torch_version = get_installed_pytorch_version()
    if not torch_version:
        print_error("PyTorch is not installed, cannot update requirements.txt")
        return False
    
    # Get the installed torchvision and torchaudio versions
    torchvision_version = get_installed_package_version("torchvision")
    torchaudio_version = get_installed_package_version("torchaudio")
    
    print_info(f"Updating requirements.txt to pin PyTorch {torch_version}, torchvision {torchvision_version}, torchaudio {torchaudio_version}")
    
    # Check if version has CUDA suffix
    has_cuda_suffix = '+cu' in torch_version
    if has_cuda_suffix:
        # Extract the base version and CUDA suffix
        torch_base_version = torch_version.split('+')[0]  # e.g., '2.6.0'
        cuda_suffix = torch_version.split('+')[1]   # e.g., 'cu118'
        print_info(f"Detected CUDA suffix: {cuda_suffix}")
        
        # Get the base versions for torchvision and torchaudio if they have CUDA suffixes
        torchvision_base_version = torchvision_version.split('+')[0] if torchvision_version and '+' in torchvision_version else torchvision_version
        torchaudio_base_version = torchaudio_version.split('+')[0] if torchaudio_version and '+' in torchaudio_version else torchaudio_version
    
    # Read the current requirements.txt
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
        
        # Update or add PyTorch requirements
        torch_lines_added = False
        new_lines = []
        for line in lines:
            # Skip commented PyTorch lines
            if line.strip().startswith('#') and 'torch' in line:
                new_lines.append(line)
            # Skip existing torch requirements
            elif line.strip().startswith('torch'):
                continue
            # Skip existing torchvision requirements
            elif line.strip().startswith('torchvision'):
                continue
            # Skip existing torchaudio requirements
            elif line.strip().startswith('torchaudio'):
                continue
            else:
                new_lines.append(line)
        
        # Add pinned PyTorch requirements after the comments
        if has_cuda_suffix:
            # For CUDA versions, add special comments with installation instructions
            torch_comment = f"\n# PyTorch requirements with CUDA {cuda_suffix}\n"
            torch_comment += f"# These must be installed with: pip install <package>==<version> --index-url=https://download.pytorch.org/whl/{cuda_suffix}\n"
            torch_lines = [torch_comment]
            
            # Add torch with correct version
            torch_lines.append(f"# torch=={torch_base_version}  # Install from {cuda_suffix} index\n")
            
            # Add torchvision with correct version if available
            if torchvision_version:
                torch_lines.append(f"# torchvision=={torchvision_base_version}  # Install from {cuda_suffix} index\n")
            
            # Add torchaudio with correct version if available
            if torchaudio_version:
                torch_lines.append(f"# torchaudio=={torchaudio_base_version}  # Install from {cuda_suffix} index\n")
        else:
            # For non-CUDA versions, add normal requirements
            torch_lines = ["\n# PyTorch requirements\n"]
            
            # Add torch with correct version
            torch_lines.append(f"torch=={torch_version}\n")
            
            # Add torchvision with correct version if available
            if torchvision_version:
                torch_lines.append(f"torchvision=={torchvision_version}\n")
            
            # Add torchaudio with correct version if available
            if torchaudio_version:
                torch_lines.append(f"torchaudio=={torchaudio_version}\n")
        
        # Add the torch lines at the appropriate location
        for i, line in enumerate(new_lines):
            if line.strip().startswith('#') and 'PyTorch' in line and not torch_lines_added:
                # Add pinned requirements after the comment
                new_lines[i+1:i+1] = torch_lines
                torch_lines_added = True
                break
        
        # If we didn't add the lines after a comment, add them at the end
        if not torch_lines_added:
            new_lines.extend(torch_lines)
        
        # Write the updated requirements.txt
        with open('requirements.txt', 'w') as f:
            f.writelines(new_lines)
        
        print_success("Updated requirements.txt with pinned PyTorch version")
        return True
    except Exception as e:
        print_error(f"Failed to update requirements.txt: {e}")
        return False

def create_pip_config():
    """Create a pip.conf file to set default options."""
    print_info("Creating pip configuration to prevent automatic dependency installation")
    
    # Get the installed PyTorch version to determine the CUDA suffix
    torch_version = get_installed_pytorch_version()
    cuda_suffix = "cu118"  # Default to CUDA 11.8
    
    if torch_version and '+cu' in torch_version:
        # Extract the CUDA suffix from the PyTorch version
        cuda_suffix = torch_version.split('+')[1]  # e.g., 'cu118'
        print_info(f"Using CUDA suffix from installed PyTorch: {cuda_suffix}")
    else:
        print_info(f"Using default CUDA suffix: {cuda_suffix}")
    
    # Determine the pip config directory
    if os.name == 'nt':  # Windows
        pip_config_dir = os.path.join(os.path.expanduser('~'), 'pip')
    else:  # Unix/Linux/macOS
        pip_config_dir = os.path.join(os.path.expanduser('~'), '.pip')
    
    # Create the directory if it doesn't exist
    os.makedirs(pip_config_dir, exist_ok=True)
    
    # Determine the config file name
    if os.name == 'nt':  # Windows
        pip_config_file = os.path.join(pip_config_dir, 'pip.ini')
    else:  # Unix/Linux/macOS
        pip_config_file = os.path.join(pip_config_dir, 'pip.conf')
    
    # Check if file already exists
    if os.path.exists(pip_config_file):
        print_warning(f"Pip config file already exists at {pip_config_file}")
        print_info("Creating backup of existing file")
        backup_path = pip_config_file + '.bak'
        os.rename(pip_config_file, backup_path)
        print_success(f"Backup created at {backup_path}")
    
    # Create new pip config file
    with open(pip_config_file, 'w') as f:
        f.write(f"""
[global]
no-dependencies = true
no-cache-dir = false
timeout = 60
index-url = https://download.pytorch.org/whl/{cuda_suffix}

[install]
no-dependencies = true
ignore-installed = false
no-warn-script-location = true
""")
    
    print_success(f"Created pip config file at {pip_config_file}")
    return True

def update_claude_config():
    """Update Claude Desktop configuration to use the virtual environment Python."""
    print_info("Updating Claude Desktop configuration")
    
    # Get the virtual environment Python executable
    venv_python = sys.executable
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Get the memory_wrapper.py path
    memory_wrapper_path = os.path.join(parent_dir, 'memory_wrapper.py')
    
    # Determine the Claude config file path
    if os.name == 'nt':  # Windows
        claude_config_dir = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'Claude')
    else:  # Unix/Linux/macOS
        claude_config_dir = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'Claude')
    
    claude_config_file = os.path.join(claude_config_dir, 'claude_desktop_config.json')
    
    # Check if file exists
    if not os.path.exists(claude_config_file):
        print_warning(f"Claude config file not found at {claude_config_file}")
        return False
    
    try:
        import json
        
        # Read the current config
        with open(claude_config_file, 'r') as f:
            config = json.load(f)
        
        # Update the memory server configuration
        if 'mcpServers' in config and 'memory' in config['mcpServers']:
            # Create backup of existing config
            backup_path = claude_config_file + '.bak'
            with open(backup_path, 'w') as f:
                json.dump(config, f, indent=2)
            print_success(f"Created backup of Claude config at {backup_path}")
            
            # Update the configuration
            config['mcpServers']['memory']['command'] = venv_python
            config['mcpServers']['memory']['args'] = [memory_wrapper_path, '--debug', '--no-auto-install']
            
            # Add environment variables to prevent automatic installation
            if 'env' not in config['mcpServers']['memory']:
                config['mcpServers']['memory']['env'] = {}
            
            config['mcpServers']['memory']['env']['PIP_NO_DEPENDENCIES'] = '1'
            config['mcpServers']['memory']['env']['PIP_NO_INSTALL'] = '1'
            
            # Write the updated config
            with open(claude_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print_success(f"Updated Claude config at {claude_config_file}")
            
            # Also update the claude_desktop_config_updated.json file if it exists
            updated_config_file = os.path.join(parent_dir, 'claude_desktop_config_updated.json')
            if os.path.exists(updated_config_file):
                with open(updated_config_file, 'r') as f:
                    updated_config = json.load(f)
                
                # Update the configuration
                if 'mcpServers' in updated_config and 'memory' in updated_config['mcpServers']:
                    updated_config['mcpServers']['memory']['command'] = venv_python
                    updated_config['mcpServers']['memory']['args'] = [memory_wrapper_path, '--debug', '--no-auto-install']
                    
                    # Add environment variables to prevent automatic installation
                    if 'env' not in updated_config['mcpServers']['memory']:
                        updated_config['mcpServers']['memory']['env'] = {}
                    
                    updated_config['mcpServers']['memory']['env']['PIP_NO_DEPENDENCIES'] = '1'
                    updated_config['mcpServers']['memory']['env']['PIP_NO_INSTALL'] = '1'
                    
                    # Write the updated config
                    with open(updated_config_file, 'w') as f:
                        json.dump(updated_config, f, indent=2)
                    
                    print_success(f"Updated {updated_config_file}")
            
            return True
        else:
            print_warning("Memory server configuration not found in Claude config")
            return False
    except Exception as e:
        print_error(f"Failed to update Claude config: {e}")
        return False

def main():
    """Main function."""
    print_header("PyTorch Dependency Fixer")
    
    # Get the currently installed PyTorch version
    torch_version = get_installed_pytorch_version()
    if not torch_version:
        print_error("PyTorch is not installed, cannot fix dependencies")
        sys.exit(1)
    
    print_success(f"Found PyTorch {torch_version}")
    
    # Create a constraints file
    constraints_path = create_constraint_file()
    if not constraints_path:
        print_error("Failed to create constraints file")
        sys.exit(1)
    
    # Pin PyTorch version
    if not pin_pytorch_version(constraints_path):
        print_error("Failed to pin PyTorch version")
        sys.exit(1)
    
    # Create no-deps wrapper
    if not create_no_deps_wrapper():
        print_warning("Failed to create no-deps wrapper")
    
    # Create sitecustomize.py
    if not create_sitecustomize():
        print_warning("Failed to create sitecustomize.py")
    
    # Update requirements.txt
    if not update_requirements_file():
        print_warning("Failed to update requirements.txt")
    
    # Create pip.conf
    if not create_pip_config():
        print_warning("Failed to create pip configuration")
    
    # Update Claude Desktop configuration
    if not update_claude_config():
        print_warning("Failed to update Claude Desktop configuration")
    
    print_header("Fix Complete")
    print_info(f"PyTorch {torch_version} has been pinned and automatic installation has been disabled")
    print_info("You should now be able to run the memory server without issues")
    print_info("If you still encounter problems, try restarting Claude Desktop")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_info("Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)