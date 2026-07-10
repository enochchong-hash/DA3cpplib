#!/bin/bash
# Compat alias for the old gemma4 script name.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stop_all.sh" "$@"
