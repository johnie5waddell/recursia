#!/bin/bash

echo "=== Backend Health Check ==="
echo

# Check if backend is running
echo "1. Checking if backend is running on port 8080..."
if nc -z localhost 8080 2>/dev/null; then
    echo "   ✓ Backend is listening on port 8080"
else
    echo "   ✗ Backend is NOT running on port 8080"
    echo "   Please start the backend with: npm run backend"
    exit 1
fi

echo
echo "2. Testing /api/health endpoint..."
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8080/api/health)
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
BODY=$(echo "$HEALTH_RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✓ Health check passed: $BODY"
else
    echo "   ✗ Health check failed with status $HTTP_CODE"
fi

echo
echo "3. Testing /api/metrics endpoint..."
METRICS_RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8080/api/metrics)
HTTP_CODE=$(echo "$METRICS_RESPONSE" | tail -n1)
BODY=$(echo "$METRICS_RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✓ Metrics endpoint working"
    echo "   Sample metrics: $(echo "$BODY" | jq -c '{rsp, coherence, entropy, state_count}' 2>/dev/null || echo "$BODY" | head -c 100)"
else
    echo "   ✗ Metrics endpoint failed with status $HTTP_CODE"
    echo "   Error: $(echo "$BODY" | jq -r '.detail' 2>/dev/null || echo "$BODY" | head -c 200)"
fi

echo
echo "4. Testing /api/states endpoint..."
STATES_RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8080/api/states)
HTTP_CODE=$(echo "$STATES_RESPONSE" | tail -n1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✓ States endpoint working"
else
    echo "   ✗ States endpoint failed with status $HTTP_CODE"
fi

echo
echo "5. Testing WebSocket connection..."
# Use timeout to limit connection test
timeout 2 bash -c 'exec 3<>/dev/tcp/localhost/8080 && echo "   ✓ WebSocket port is accessible"' 2>/dev/null || echo "   ✗ WebSocket connection failed"

echo
echo "=== Health Check Complete ==="