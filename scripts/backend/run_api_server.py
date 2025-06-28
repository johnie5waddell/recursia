#!/usr/bin/env python3
"""
Direct API Server Runner
Runs the Unified API Server with centralized runtime
"""

import sys
import os
import asyncio
import uvicorn

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import unified API server
from src.api.unified_api_server import UnifiedAPIServer

def main():
    """Run the unified API server."""
    # Create server instance
    server = UnifiedAPIServer(debug=False)
    
    # Get port from environment variables with fallbacks
    port = int(os.environ.get('RECURSIA_BACKEND_PORT', os.environ.get('PORT', 8080)))
    
    # Get host configuration
    host = os.environ.get('RECURSIA_HOST', '127.0.0.1')
    
    # Get log level
    log_level = os.environ.get('RECURSIA_LOG_LEVEL', 'info').lower()
    
    # Run the server
    print(f"Starting server on {host}:{port}")
    print(f"Log level: {log_level}")
    
    uvicorn.run(
        server.app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    print("Starting Recursia Unified API Server...")
    main()