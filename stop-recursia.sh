#!/bin/bash
# Convenience wrapper to call the actual stop script

# Get the directory of this script (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call the actual stop script
exec "$PROJECT_ROOT/scripts/deployment/stop-recursia.sh" "$@"