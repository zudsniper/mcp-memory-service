# MCP Memory Service - Fixed Configuration

This document provides instructions for fixing the PyTorch dependency issues and MCP protocol compatibility in the MCP Memory Service for Windows.

## Problem Description

Two main issues were identified and fixed:

1. **PyTorch Dependency Issue**: The original configuration was attempting to install PyTorch 2.5.1 from PyPI, which fails on Windows because PyTorch requires a special index URL for Windows installations.

2. **MCP Protocol Compatibility**: The server was not implementing required MCP protocol methods, resulting in "Method not found" errors and JSON error popups in Claude Desktop.

## Solution

We've created several scripts and made code changes to fix these issues:

1. `scripts/fix_sitecustomize.py` - Fixes the sitecustomize.py file to prevent recursion issues
2. `scripts/run_memory_server.py` - Directly runs the memory server without going through the installation process
3. `claude_desktop_config_updated.json` - Updated configuration for Claude Desktop
4. Updated `src/mcp_memory_service/server.py` - Added missing MCP protocol methods

## How to Use

1. First, run the sitecustomize fix script to prevent recursion issues:

```bash
venv\Scripts\python.exe scripts\fix_sitecustomize.py
```

2. Update your Claude Desktop configuration:

Use the `claude_desktop_config_template.json` file as a template for your Claude Desktop configuration:

1. Copy the template file:
   ```bash
   copy claude_desktop_config_template.json %APPDATA%\Claude\claude_desktop_config.json
   ```

2. Edit the configuration file to replace the placeholders with your actual paths:
   - `${PYTHON_PATH}` - Path to your Python executable (e.g., `C:\Users\username\mcp-memory-service\venv\Scripts\python.exe`)
   - `${PROJECT_PATH}` - Path to your project directory (e.g., `C:\Users\username\mcp-memory-service`)
   - `${USER_DATA_PATH}` - Path to your user data directory (e.g., `C:\Users\username\AppData\Local`)

3. Restart Claude Desktop

After updating the configuration, restart Claude Desktop to apply the changes.

## Troubleshooting

If you encounter issues:

1. Check the logs in `%APPDATA%\Claude\logs\mcp-server-memory.log`
2. Run the memory server directly to see any errors:

```bash
venv\Scripts\python.exe scripts\run_memory_server.py
```

3. Verify PyTorch is working correctly:

```bash
venv\Scripts\python.exe scripts\verify_torch.py
```

## Technical Details

### PyTorch Dependency Fix:

1. **Recursion Issue**: The sitecustomize.py file was causing a recursion error when trying to import PyTorch.
2. **Import Hooks**: We disabled import hooks that were causing issues with PyTorch imports.
3. **Direct Module Loading**: We implemented direct module loading using importlib to bypass the problematic import system.
4. **Environment Variables**: We set environment variables to prevent automatic dependency installation.

### MCP Protocol Compatibility Fix:

1. **Missing Methods**: We implemented the required MCP protocol methods (resources/list, resources/read, resource_templates/list).
2. **Custom Error Handler**: We added a custom error handler for unsupported methods.
3. **Type Annotations**: We fixed type annotations to match the actual types used in the MCP library.

See `MCP_PROTOCOL_FIX.md` for detailed information about the MCP protocol compatibility fix.

### Configuration Changes:

The updated Claude Desktop configuration:
- Uses the direct runner script instead of the memory_wrapper.py
- Sets environment variables to prevent automatic dependency installation
- Uses the virtual environment Python executable

## Additional Scripts

- `scripts/fix_pytorch_dependency.py` - Pins PyTorch to the currently installed version
- `scripts/verify_torch.py` - Verifies PyTorch is working correctly
- `scripts/debug_dependencies.py` - Provides detailed information about installed dependencies