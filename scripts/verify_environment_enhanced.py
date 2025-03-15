#!/usr/bin/env python3
"""
Enhanced environment verification script for MCP Memory Service.
This script checks the system environment, hardware capabilities,
and installed dependencies to ensure compatibility.
"""
import os
import sys
import platform
import subprocess
import json
import importlib
import pkg_resources
from pathlib import Path
import traceback

class EnvironmentVerifier:
    def __init__(self):
        self.verification_results = []
        self.critical_failures = []
        self.warnings = []
        self.system_info = self.detect_system()
        self.gpu_info = self.detect_gpu()
        self.claude_config = self.load_claude_config()
        
    def detect_system(self):
        """Detect system architecture and platform."""
        system_info = {
            "os_name": platform.system().lower(),
            "os_version": platform.version(),
            "architecture": platform.machine().lower(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count() or 1,
            "memory_gb": self.get_system_memory(),
            "in_virtual_env": sys.prefix != sys.base_prefix
        }
        
        self.verification_results.append(
            f"✓ System: {platform.system()} {platform.version()}"
        )
        self.verification_results.append(
            f"✓ Architecture: {system_info['architecture']}"
        )
        self.verification_results.append(
            f"✓ Python: {system_info['python_version']}"
        )
        
        if system_info["in_virtual_env"]:
            self.verification_results.append(
                f"✓ Virtual environment: {sys.prefix}"
            )
        else:
            self.warnings.append(
                "Not running in a virtual environment"
            )
        
        return system_info
    
    def get_system_memory(self):
        """Get the total system memory in GB."""
        try:
            if self.system_info["os_name"] == "linux":
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            # Extract the memory value (in kB)
                            memory_kb = int(line.split()[1])
                            return round(memory_kb / (1024 * 1024), 2)  # Convert to GB
                            
            elif self.system_info["os_name"] == "darwin":  # macOS
                output = subprocess.check_output(['sysctl', '-n', 'hw.memsize']).decode('utf-8').strip()
                memory_bytes = int(output)
                return round(memory_bytes / (1024**3), 2)  # Convert to GB
                
            elif self.system_info["os_name"] == "windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', c_ulonglong),
                        ('ullAvailPhys', c_ulonglong),
                        ('ullTotalPageFile', c_ulonglong),
                        ('ullAvailPageFile', c_ulonglong),
                        ('ullTotalVirtual', c_ulonglong),
                        ('ullAvailVirtual', c_ulonglong),
                        ('ullAvailExtendedVirtual', c_ulonglong),
                    ]
                    
                memoryStatus = MEMORYSTATUSEX()
                memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
                return round(memoryStatus.ullTotalPhys / (1024**3), 2)  # Convert to GB
                
        except Exception as e:
            self.warnings.append(f"Failed to get system memory: {e}")
            
        # Default fallback
        return 4.0  # Assume 4GB as a conservative default
    
    def detect_gpu(self):
        """Detect GPU and acceleration capabilities."""
        gpu_info = {
            "has_cuda": False,
            "cuda_version": None,
            "has_rocm": False,
            "rocm_version": None,
            "has_mps": False,
            "has_directml": False,
            "accelerator": "cpu"  # Default to CPU
        }
        
        # Check for CUDA
        if self.system_info["os_name"] == "windows":
            cuda_path = os.environ.get('CUDA_PATH')
            if cuda_path and os.path.exists(cuda_path):
                gpu_info["has_cuda"] = True
                try:
                    # Try to get CUDA version
                    nvcc_output = subprocess.check_output([os.path.join(cuda_path, 'bin', 'nvcc'), '--version'], 
                                                         stderr=subprocess.STDOUT, 
                                                         universal_newlines=True)
                    for line in nvcc_output.split('\n'):
                        if 'release' in line:
                            gpu_info["cuda_version"] = line.split('release')[-1].strip().split(',')[0].strip()
                            break
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
        elif self.system_info["os_name"] == "linux":
            cuda_paths = ['/usr/local/cuda', os.environ.get('CUDA_HOME')]
            for path in cuda_paths:
                if path and os.path.exists(path):
                    gpu_info["has_cuda"] = True
                    try:
                        # Try to get CUDA version
                        nvcc_output = subprocess.check_output([os.path.join(path, 'bin', 'nvcc'), '--version'], 
                                                             stderr=subprocess.STDOUT, 
                                                             universal_newlines=True)
                        for line in nvcc_output.split('\n'):
                            if 'release' in line:
                                gpu_info["cuda_version"] = line.split('release')[-1].strip().split(',')[0].strip()
                                break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        pass
                    break
        
        # Check for ROCm (AMD)
        if self.system_info["os_name"] == "linux":
            rocm_paths = ['/opt/rocm', os.environ.get('ROCM_HOME')]
            for path in rocm_paths:
                if path and os.path.exists(path):
                    gpu_info["has_rocm"] = True
                    try:
                        # Try to get ROCm version
                        with open(os.path.join(path, 'bin', '.rocmversion'), 'r') as f:
                            gpu_info["rocm_version"] = f.read().strip()
                    except (FileNotFoundError, IOError):
                        try:
                            rocm_output = subprocess.check_output(['rocminfo'], 
                                                                stderr=subprocess.STDOUT, 
                                                                universal_newlines=True)
                            for line in rocm_output.split('\n'):
                                if 'Version' in line:
                                    gpu_info["rocm_version"] = line.split(':')[-1].strip()
                                    break
                        except (subprocess.SubprocessError, FileNotFoundError):
                            pass
                    break
        
        # Check for MPS (Apple Silicon)
        if self.system_info["os_name"] == "darwin" and self.system_info["architecture"] in ("arm64", "aarch64"):
            try:
                # Check if Metal is supported
                result = subprocess.run(
                    ['system_profiler', 'SPDisplaysDataType'],
                    capture_output=True,
                    text=True
                )
                gpu_info["has_mps"] = 'Metal' in result.stdout
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        
        # Check for DirectML (Windows)
        if self.system_info["os_name"] == "windows":
            try:
                # Check if DirectML package is installed
                import pkg_resources
                pkg_resources.get_distribution('torch-directml')
                gpu_info["has_directml"] = True
            except (ImportError, pkg_resources.DistributionNotFound):
                # Check if DirectML is available on the system
                try:
                    import ctypes
                    ctypes.WinDLL('DirectML.dll')
                    gpu_info["has_directml"] = True
                except (ImportError, OSError):
                    pass
        
        # Set accelerator type
        if gpu_info["has_cuda"]:
            gpu_info["accelerator"] = "cuda"
            self.verification_results.append(
                f"✓ CUDA detected: {gpu_info['cuda_version'] or 'Unknown version'}"
            )
        elif gpu_info["has_rocm"]:
            gpu_info["accelerator"] = "rocm"
            self.verification_results.append(
                f"✓ ROCm detected: {gpu_info['rocm_version'] or 'Unknown version'}"
            )
        elif gpu_info["has_mps"]:
            gpu_info["accelerator"] = "mps"
            self.verification_results.append(
                "✓ Apple Metal Performance Shaders (MPS) detected"
            )
        elif gpu_info["has_directml"]:
            gpu_info["accelerator"] = "directml"
            self.verification_results.append(
                "✓ DirectML detected"
            )
        else:
            self.verification_results.append(
                "✓ Using CPU-only mode (no GPU acceleration detected)"
            )
        
        return gpu_info

    def load_claude_config(self):
        """Load configuration from Claude Desktop config."""
        try:
            # Check common locations for Claude Desktop config
            home_dir = Path.home()
            possible_paths = [
                home_dir / "Library/Application Support/Claude/claude_desktop_config.json",
                home_dir / ".config/Claude/claude_desktop_config.json",
                Path(__file__).parent.parent / "claude_config/claude_desktop_config.json"
            ]

            for config_path in possible_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                        self.verification_results.append(
                            f"✓ Found Claude Desktop config at {config_path}"
                        )
                        return config

            self.warnings.append(
                "Could not find Claude Desktop config file in any standard location"
            )
            return None

        except Exception as e:
            self.warnings.append(
                f"Error loading Claude Desktop config: {str(e)}"
            )
            return None

    def verify_python_version(self):
        """Verify Python interpreter version matches production requirements."""
        try:
            python_version = sys.version.split()[0]
            required_version = "3.10"  # Minimum required version
            
            if not python_version.startswith(required_version) and float(python_version.split('.')[0] + '.' + python_version.split('.')[1]) < float(required_version):
                self.critical_failures.append(
                    f"Python version too old: Found {python_version}, required {required_version} or newer"
                )
            else:
                self.verification_results.append(
                    f"✓ Python version verified: {python_version}"
                )
        except Exception as e:
            self.critical_failures.append(f"Failed to verify Python version: {str(e)}")

    def verify_virtual_environment(self):
        """Verify we're running in a virtual environment."""
        try:
            if sys.prefix == sys.base_prefix:
                self.warnings.append(
                    "Not running in a virtual environment!"
                )
            else:
                self.verification_results.append(
                    f"✓ Virtual environment verified: {sys.prefix}"
                )
        except Exception as e:
            self.warnings.append(
                f"Failed to verify virtual environment: {str(e)}"
            )

    def verify_critical_packages(self):
        """Verify critical packages are installed with correct versions."""
        required_packages = {
            'chromadb': '0.5.23',
            'sentence-transformers': '2.2.2',
            'torch': '2.0.0',
            'mcp': '1.0.0'
        }

        for package, min_version in required_packages.items():
            try:
                installed_version = pkg_resources.get_distribution(package).version
                if min_version:
                    # Compare versions
                    installed_parts = [int(x) for x in installed_version.split('.')]
                    required_parts = [int(x) for x in min_version.split('.')]
                    
                    # Pad with zeros if needed
                    while len(installed_parts) < len(required_parts):
                        installed_parts.append(0)
                    while len(required_parts) < len(installed_parts):
                        required_parts.append(0)
                    
                    # Compare version components
                    for i in range(len(required_parts)):
                        if installed_parts[i] < required_parts[i]:
                            self.critical_failures.append(
                                f"Package version too old: {package} "
                                f"(found {installed_version}, required {min_version} or newer)"
                            )
                            break
                        elif installed_parts[i] > required_parts[i]:
                            # Higher version is fine
                            break
                
                self.verification_results.append(
                    f"✓ Package verified: {package} {installed_version}"
                )
            except pkg_resources.DistributionNotFound:
                self.critical_failures.append(f"Required package not found: {package}")
            except Exception as e:
                self.critical_failures.append(
                    f"Failed to verify package {package}: {str(e)}"
                )

    def verify_claude_paths(self):
        """Verify paths from Claude Desktop config."""
        if not self.claude_config:
            return

        try:
            # Extract paths from config
            mcp_memory_config = self.claude_config.get('mcp-memory', {})
            chroma_path = mcp_memory_config.get('chroma_db')
            backup_path = mcp_memory_config.get('backup_path')

            if chroma_path:
                os.environ['MCP_MEMORY_CHROMA_PATH'] = str(chroma_path)
                self.verification_results.append(
                    f"✓ Set MCP_MEMORY_CHROMA_PATH from config: {chroma_path}"
                )
            else:
                self.warnings.append("MCP_MEMORY_CHROMA_PATH not found in Claude config")

            if backup_path:
                os.environ['MCP_MEMORY_BACKUP_PATH'] = str(backup_path)
                self.verification_results.append(
                    f"✓ Set MCP_MEMORY_BACKUP_PATH from config: {backup_path}"
                )
            else:
                self.warnings.append("MCP_MEMORY_BACKUP_PATH not found in Claude config")

        except Exception as e:
            self.warnings.append(f"Failed to verify Claude paths: {str(e)}")

    def verify_import_functionality(self):
        """Verify critical imports work correctly."""
        critical_imports = [
            'chromadb',
            'sentence_transformers',
            'torch',
            'mcp',
            'mcp_memory_service'
        ]

        for module_name in critical_imports:
            try:
                module = importlib.import_module(module_name)
                self.verification_results.append(f"✓ Successfully imported {module_name}")
                
                # Additional checks for specific modules
                if module_name == 'torch':
                    # Check for CUDA
                    if hasattr(module, 'cuda') and module.cuda.is_available():
                        self.verification_results.append(f"✓ PyTorch CUDA is available: {module.version.cuda}")
                        try:
                            self.verification_results.append(f"✓ GPU: {module.cuda.get_device_name(0)}")
                        except Exception:
                            pass
                    # Check for MPS (Apple Silicon)
                    elif hasattr(module.backends, 'mps') and module.backends.mps.is_available():
                        self.verification_results.append(f"✓ PyTorch MPS is available")
                    # Check for DirectML
                    else:
                        try:
                            import torch_directml
                            self.verification_results.append(f"✓ PyTorch DirectML is available")
                        except ImportError:
                            self.verification_results.append(f"✓ PyTorch CPU-only mode")
                
            except ImportError as e:
                self.critical_failures.append(f"Failed to import {module_name}: {str(e)}")
            except Exception as e:
                self.critical_failures.append(f"Error with {module_name}: {str(e)}")

    def verify_paths(self):
        """Verify critical paths exist and are accessible."""
        critical_paths = [
            os.environ.get('MCP_MEMORY_CHROMA_PATH', ''),
            os.environ.get('MCP_MEMORY_BACKUP_PATH', '')
        ]

        for path in critical_paths:
            if not path:
                continue
            try:
                path_obj = Path(path)
                if not path_obj.exists():
                    self.warnings.append(f"Critical path does not exist: {path}")
                elif not os.access(path, os.R_OK | os.W_OK):
                    self.warnings.append(f"Insufficient permissions for path: {path}")
                else:
                    self.verification_results.append(f"✓ Path verified: {path}")
            except Exception as e:
                self.warnings.append(f"Failed to verify path {path}: {str(e)}")

    def verify_embedding_model(self):
        """Verify embedding model can be loaded and used."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try to load the model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Try to encode a test sentence
            test_embedding = model.encode("This is a test sentence.")
            
            self.verification_results.append(
                f"✓ Successfully loaded and tested embedding model: all-MiniLM-L6-v2"
            )
            self.verification_results.append(
                f"✓ Embedding dimension: {len(test_embedding)}"
            )
            
        except Exception as e:
            self.critical_failures.append(
                f"Failed to load or use embedding model: {str(e)}"
            )
            self.critical_failures.append(traceback.format_exc())

    def verify_chromadb(self):
        """Verify ChromaDB can be initialized and used."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            # Create a temporary directory for testing
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize ChromaDB client
                client = chromadb.PersistentClient(path=temp_dir)
                
                # Create embedding function
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name='all-MiniLM-L6-v2'
                )
                
                # Create a test collection
                collection = client.create_collection(
                    name="test_collection",
                    embedding_function=ef
                )
                
                # Add a test document
                collection.add(
                    documents=["This is a test document."],
                    metadatas=[{"source": "test"}],
                    ids=["test1"]
                )
                
                # Query the collection
                results = collection.query(
                    query_texts=["test document"],
                    n_results=1
                )
                
                if results and results["ids"] and results["ids"][0]:
                    self.verification_results.append(
                        f"✓ Successfully tested ChromaDB functionality"
                    )
                else:
                    self.warnings.append(
                        "ChromaDB query returned no results"
                    )
                
        except Exception as e:
            self.critical_failures.append(
                f"Failed to initialize or use ChromaDB: {str(e)}"
            )
            self.critical_failures.append(traceback.format_exc())

    def run_verifications(self):
        """Run all verifications."""
        self.verify_python_version()
        self.verify_virtual_environment()
        self.verify_critical_packages()
        self.verify_claude_paths()
        self.verify_import_functionality()
        self.verify_paths()
        
        # More intensive tests
        print("\nRunning intensive tests (may take a moment)...")
        self.verify_embedding_model()
        self.verify_chromadb()

    def print_results(self):
        """Print verification results."""
        print("\n=== Environment Verification Results ===\n")
        
        if self.verification_results:
            print("Successful Verifications:")
            for result in self.verification_results:
                print(f"  {result}")
        
        if self.warnings:
            print("\nWarnings (non-critical issues):")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
        
        if self.critical_failures:
            print("\nCritical Failures:")
            for failure in self.critical_failures:
                print(f"  ✗ {failure}")
        
        print("\nSummary:")
        print(f"  Passed: {len(self.verification_results)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Failed: {len(self.critical_failures)}")
        
        if self.critical_failures:
            print("\nTo fix these issues:")
            print("1. Create a new virtual environment:")
            print("   python -m venv venv")
            print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            print("\n2. Install the package with the installation script:")
            print("   python install.py")
            print("\n3. Ensure Claude Desktop config is properly set up with required paths")
            
        return len(self.critical_failures) == 0

    def export_diagnostics(self, output_file=None):
        """Export diagnostics to a JSON file."""
        if not output_file:
            output_file = "mcp_memory_diagnostics.json"
        
        diagnostics = {
            "system_info": self.system_info,
            "gpu_info": self.gpu_info,
            "verification_results": self.verification_results,
            "warnings": self.warnings,
            "critical_failures": self.critical_failures,
            "environment_variables": {k: v for k, v in os.environ.items() if k.startswith(('MCP_', 'PYTORCH_', 'CUDA_', 'ROCM_'))},
            "python_path": sys.executable,
            "python_version": sys.version,
            "timestamp": import_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add installed packages
        try:
            diagnostics["installed_packages"] = {
                pkg.key: pkg.version for pkg in pkg_resources.working_set
            }
        except Exception as e:
            diagnostics["installed_packages_error"] = str(e)
        
        # Write to file
        try:
            with open(output_file, 'w') as f:
                json.dump(diagnostics, f, indent=2)
            print(f"\nDiagnostics exported to: {output_file}")
        except Exception as e:
            print(f"\nFailed to export diagnostics: {str(e)}")

def main():
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Verify MCP Memory Service environment")
    parser.add_argument("--export", action="store_true", help="Export diagnostics to a JSON file")
    parser.add_argument("--output", type=str, help="Output file for diagnostics (default: mcp_memory_diagnostics.json)")
    args = parser.parse_args()
    
    print("=== MCP Memory Service Environment Verification ===")
    print("Running comprehensive environment checks...")
    
    verifier = EnvironmentVerifier()
    verifier.run_verifications()
    environment_ok = verifier.print_results()
    
    if args.export:
        verifier.export_diagnostics(args.output)
    
    if not environment_ok:
        print("\n⚠️  Environment verification failed! Service may not function correctly.")
        sys.exit(1)
    else:
        print("\n✓ Environment verification passed! Service should function correctly.")
        sys.exit(0)

if __name__ == "__main__":
    main()