# MCP Memory Service Documentation

Welcome to the MCP Memory Service documentation. This directory contains comprehensive guides for installing, using, and troubleshooting the service.

## Guides

- [Installation Guide](guides/installation.md) - Comprehensive installation instructions for all platforms
- [Troubleshooting Guide](guides/troubleshooting.md) - Solutions for common issues and debugging procedures

## Technical Documentation

- [Tag Storage Procedure](technical/tag-storage.md) - Technical details about tag storage and migration

## Quick Links

- [Main README](../README.md) - Overview of the service and its features
- [Hardware Compatibility](../README.md#hardware-compatibility) - Supported platforms and accelerators
- [Configuration Options](../README.md#configuration-options) - Available environment variables and settings

## Platform-Specific Notes

### Windows
- Uses Windows-specific installation script
- Requires PyTorch wheels from specific index URL
- See [Windows Installation Guide](guides/installation.md#windows)

### macOS
- Intel x86_64: Uses specific PyTorch versions for compatibility
- Apple Silicon: Supports MPS acceleration with fallbacks
- See [macOS Installation Guide](guides/installation.md#macos)

## Available Scripts

The `scripts/` directory contains several useful tools:
- `run_memory_server.py` - Direct runner for MCP Memory Service
- `verify_environment_enhanced.py` - Environment compatibility verification
- `install_windows.py` - Windows-specific installation
- `fix_sitecustomize.py` - Fix for recursion issues
- `verify_pytorch_windows.py` - Windows PyTorch verification

## Configuration

- See [Claude MCP Configuration](../README.md#claude-mcp-configuration) for configuration options
- Sample configuration templates are available in the `claude_config/` directory