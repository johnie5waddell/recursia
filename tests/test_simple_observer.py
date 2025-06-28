import requests
import json

# First, check current observers
response = requests.get('http://localhost:8080/api/observers')
print("Current observers:")
print(json.dumps(response.json(), indent=2))

# Simple code with one observer
code = '''
state TestState : quantum_type {
    state_qubits: 1
}

observer SimpleObserver {
    observer_type: "standard_observer"
}

print "Observer created";
'''

response = requests.post(
    'http://localhost:8080/api/execute',
    json={'code': code}
)

print("\nExecution result:")
print(json.dumps(response.json(), indent=2))

# Check observers again
response = requests.get('http://localhost:8080/api/observers')
print("\nObservers after execution:")
print(json.dumps(response.json(), indent=2))

# Check metrics
response = requests.get('http://localhost:8080/api/metrics')
metrics = response.json()
print(f"\nMetrics observer_count: {metrics.get('observer_count', 0)}")
print(f"Metrics state_count: {metrics.get('state_count', 0)}")