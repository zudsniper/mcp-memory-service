import { MCPServer } from '@anthropic-dev/mcp-server';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class MemoryServer extends MCPServer {
  constructor() {
    super('memory-server');
    this.pythonProcess = null;
    this.initializePythonBackend();
  }

  async initializePythonBackend() {
    // Start the Python backend
    const pythonScript = path.join(__dirname, 'memory_backend.py');
    this.pythonProcess = spawn('python', [pythonScript]);

    this.pythonProcess.stdout.on('data', (data) => {
      console.log(`Python backend: ${data}`);
    });

    this.pythonProcess.stderr.on('data', (data) => {
      console.error(`Python backend error: ${data}`);
    });

    // Clean up on exit
    process.on('exit', () => {
      if (this.pythonProcess) {
        this.pythonProcess.kill();
      }
    });
  }
}

const server = new MemoryServer();
server.start();