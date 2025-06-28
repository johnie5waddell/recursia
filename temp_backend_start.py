import sys
import traceback

try:
    print("Starting backend server...", flush=True)
    sys.path.insert(0, '.')
    from src.api.unified_api_server import app
    import uvicorn
    
    print(f"Python path: {sys.path}", flush=True)
    print("Starting uvicorn...", flush=True)
    
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
except Exception as e:
    print(f"Backend error: {str(e)}", flush=True)
    print(traceback.format_exc(), flush=True)
    sys.exit(1)
