#!/bin/bash

# Start dev server in background
npm run dev &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Test if server is responding
echo "Testing server at http://localhost:3001..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/)

if [ "$RESPONSE" == "200" ]; then
    echo "✅ Frontend is loading successfully! HTTP 200 OK"
else
    echo "❌ Frontend returned HTTP $RESPONSE"
fi

# Kill the server
kill $SERVER_PID