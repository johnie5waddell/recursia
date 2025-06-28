#!/bin/bash
# Convenience wrapper to call the actual startup script

# Get the directory of this script (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call the actual startup script
exec "$PROJECT_ROOT/scripts/deployment/start-recursia.sh" "$@"