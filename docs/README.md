# MCP Memory Service Documentation

Welcome to the MCP Memory Service documentation. This directory contains comprehensive guides for installing, using, and troubleshooting the service.

## Available Guides

- [Installation Guide](guides/installation.md) - Detailed installation instructions for all platforms
- [Troubleshooting](guides/installation.md#troubleshooting-common-installation-issues) - Solutions to common installation and usage issues

## Quick Links

- [Main README](../README.md) - Overview of the service and its features
- [CLAUDE.md](../CLAUDE.md) - Development guidelines for AI assistants
- [MCP Protocol Fix](../MCP_PROTOCOL_FIX.md) - Details about MCP protocol compatibility fixes

## Platform-Specific Installation Notes

### macOS (Intel x86_64)

For macOS running on Intel (x86_64) processors, the following PyTorch versions are required:
- torch==1.13.1
- torchvision==0.14.1 
- torchaudio==0.13.1

You can install these with:
```bash
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install sentence-transformers==2.2.2
```

Our installation scripts (`install.py` and `memory_wrapper.py`) now automatically detect macOS on Intel and will install the correct versions. If you were previously experiencing issues with the Windows-specific installation script being erroneously used on macOS, this has been fixed.

For more details, see the [macOS installation section](guides/installation.md#intel-cpus) in the installation guide.

### Windows

Windows requires the Windows-specific installation script and PyTorch wheels from a specific index URL:
```bash
python scripts/install_windows.py
```

### Apple Silicon

Apple Silicon Macs work best with:
- torch==2.1.0
- torchvision==2.1.0
- torchaudio==2.1.0

## Scripts

The `scripts/` directory contains several useful tools:

- `run_memory_server.py` - Direct runner for MCP Memory Service
- `verify_environment_enhanced.py` - Verify your environment compatibility
- `install_windows.py` - Windows-specific installation script
- `fix_sitecustomize.py` - Fix for recursion issues in sitecustomize.py
- `verify_pytorch_windows.py` - Verify PyTorch installation on Windows

## Configuration

- See [Claude MCP Configuration](../README.md#claude-mcp-configuration) for configuration options
- Sample configuration templates are available in the `claude_config/` directory