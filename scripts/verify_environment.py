#!/usr/bin/env python3
import sys
import os
import pkg_resources
import importlib
import platform
import json
from pathlib import Path

class EnvironmentVerifier:
    def __init__(self):
        self.verification_results = []
        self.critical_failures = []
        self.claude_config = self.load_claude_config()

    def load_claude_config(self):
        """Load configuration from Claude Desktop config."""
        try:
            # Check common locations for Claude Desktop config
            possible_paths = [
                Path.home() / "Library/Application Support/Claude Desktop/claude_desktop_config.json",
                Path.home() / ".config/Claude Desktop/claude_desktop_config.json",
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

            self.critical_failures.append(
                "Could not find Claude Desktop config file in any standard location"
            )
            return None

        except Exception as e:
            self.critical_failures.append(
                f"Error loading Claude Desktop config: {str(e)}"
            )
            return None

    def verify_python_version(self):
        """Verify Python interpreter version matches production requirements."""
        try:
            python_version = sys.version.split()[0]
            required_version = "3.9"  # Production version
            
            if not python_version.startswith(required_version):
                self.critical_failures.append(
                    f"Python version mismatch: Found {python_version}, required {required_version}"
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
                self.critical_failures.append(
                    "Not running in a virtual environment!"
                )
            else:
                self.verification_results.append(
                    f"✓ Virtual environment verified: {sys.prefix}"
                )
        except Exception as e:
            self.critical_failures.append(
                f"Failed to verify virtual environment: {str(e)}"
            )

    def verify_critical_packages(self):
        """Verify critical packages are installed with correct versions."""
        required_packages = {
            'chromadb': '0.5.23',
            'sentence-transformers': '2.2.2',
            'urllib3': '1.26.6',
            'python-dotenv': '1.0.0'
        }

        for package, required_version in required_packages.items():
            try:
                installed_version = pkg_resources.get_distribution(package).version
                if required_version and installed_version != required_version:
                    self.critical_failures.append(
                        f"Package version mismatch: {package} "
                        f"(found {installed_version}, required {required_version})"
                    )
                else:
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
            chroma_path = self.claude_config.get('mcp-memory', {}).get('chroma_db')
            backup_path = self.claude_config.get('mcp-memory', {}).get('backup_path')

            if chroma_path:
                os.environ['CHROMA_DB_PATH'] = str(chroma_path)
                self.verification_results.append(
                    f"✓ Set CHROMA_DB_PATH from config: {chroma_path}"
                )
            else:
                self.critical_failures.append("CHROMA_DB_PATH not found in Claude config")

            if backup_path:
                os.environ['MCP_MEMORY_BACKUP_PATH'] = str(backup_path)
                self.verification_results.append(
                    f"✓ Set MCP_MEMORY_BACKUP_PATH from config: {backup_path}"
                )
            else:
                self.critical_failures.append("MCP_MEMORY_BACKUP_PATH not found in Claude config")

        except Exception as e:
            self.critical_failures.append(f"Failed to verify Claude paths: {str(e)}")

    def verify_import_functionality(self):
        """Verify critical imports work correctly."""
        critical_imports = [
            'chromadb',
            'sentence_transformers',
        ]

        for module_name in critical_imports:
            try:
                module = importlib.import_module(module_name)
                self.verification_results.append(f"✓ Successfully imported {module_name}")
            except ImportError as e:
                self.critical_failures.append(f"Failed to import {module_name}: {str(e)}")

    def verify_paths(self):
        """Verify critical paths exist and are accessible."""
        critical_paths = [
            os.environ.get('CHROMA_DB_PATH', ''),
            os.environ.get('MCP_MEMORY_BACKUP_PATH', '')
        ]

        for path in critical_paths:
            if not path:
                continue
            try:
                path_obj = Path(path)
                if not path_obj.exists():
                    self.critical_failures.append(f"Critical path does not exist: {path}")
                elif not os.access(path, os.R_OK | os.W_OK):
                    self.critical_failures.append(f"Insufficient permissions for path: {path}")
                else:
                    self.verification_results.append(f"✓ Path verified: {path}")
            except Exception as e:
                self.critical_failures.append(f"Failed to verify path {path}: {str(e)}")

    def run_verifications(self):
        """Run all verifications."""
        self.verify_python_version()
        self.verify_virtual_environment()
        self.verify_critical_packages()
        self.verify_claude_paths()  # This now sets environment variables from config
        self.verify_import_functionality()
        self.verify_paths()

    def print_results(self):
        """Print verification results."""
        print("\n=== Environment Verification Results ===\n")
        
        if self.verification_results:
            print("Successful Verifications:")
            for result in self.verification_results:
                print(f"  {result}")
        
        if self.critical_failures:
            print("\nCritical Failures:")
            for failure in self.critical_failures:
                print(f"  ✗ {failure}")
        
        print("\nSummary:")
        print(f"  Passed: {len(self.verification_results)}")
        print(f"  Failed: {len(self.critical_failures)}")
        
        if self.critical_failures:
            print("\nTo fix these issues:")
            print("1. Create a new virtual environment:")
            print("   conda create -n migration-env python=3.9")
            print("   conda activate migration-env")
            print("\n2. Install requirements:")
            print("   pip install -r requirements.txt")
            print("\n3. Ensure Claude Desktop config is properly set up with required paths")
            
        return len(self.critical_failures) == 0

def main():
    verifier = EnvironmentVerifier()
    verifier.run_verifications()
    environment_ok = verifier.print_results()
    
    if not environment_ok:
        print("\n⚠️  Environment verification failed! Migration cannot proceed.")
        sys.exit(1)
    else:
        print("\n✓ Environment verification passed! Safe to proceed with migration.")
        sys.exit(0)

if __name__ == "__main__":
    main()
