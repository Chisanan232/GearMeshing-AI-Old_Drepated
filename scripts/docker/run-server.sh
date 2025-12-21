#!/bin/bash
set -e

#
# Environment variables:
#
# SERVER_HOST → --host
# SERVER_PORT → --port
# LOG_LEVEL → --log-level
# RELOAD → --reload
#

# Initialize command line arguments array
CMD_ARGS=()

# Map environment variables to command line options

# HOST: Host for FastAPI HTTP transports (used for sse or streamable-http)
if [ -n "${GEARMESHING_AI_SERVER_HOST}" ]; then
  CMD_ARGS+=(--host "${GEARMESHING_AI_SERVER_HOST}")
fi

# PORT: Port for FastAPI HTTP transports
if [ -n "${GEARMESHING_AI_SERVER_PORT}" ]; then
  CMD_ARGS+=(--port "${GEARMESHING_AI_SERVER_PORT}")
fi

# LOG_LEVEL: Python logging level
if [ -n "${GEARMESHING_AI_LOG_LEVEL}" ]; then
  CMD_ARGS+=(--log-level "${GEARMESHING_AI_LOG_LEVEL}")
fi

# RETRY: Number of retry attempts for network operations
if [ -n "${RELOAD}" ]; then
  CMD_ARGS+=(--reload "${RELOAD}")
fi

# Print the command that will be executed
echo "Starting AI agent server with arguments: ${CMD_ARGS[@]}"
# Only print debug command information if log level is debug (case insensitive)
if [ -n "${LOG_LEVEL}" ] && [ "$(echo ${LOG_LEVEL} | tr '[:upper:]' '[:lower:]')" == "debug" ]; then
  echo "[DEBUG] Run the AI agent server with command: uv run <gearmeshing-ai cli> ${CMD_ARGS[@]}"
fi

# Execute the entry point with the collected arguments
exec uv run uvicorn gearmeshing_ai.server.main:app "${CMD_ARGS[@]}"
