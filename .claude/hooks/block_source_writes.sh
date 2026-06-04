#!/usr/bin/env bash
set -euo pipefail

INPUT="$(cat)"
TOOL_NAME="$(echo "$INPUT" | jq -r 'tool_name // empty')"
FILE_PATH="$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')"
COMMAND="$(echo "$INPUT" | jq -r '.tool_input.command // empty')"

# case "$FILE_PATH" in
    # */src/*|*/incluude/*|*/cuda/*|*/tests/*|*/python/*)

case "$FILE_PATH" in
    (*/src/*|*/include/*|*/cuda/*|*/tests/*|*/python/*)
        echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Source tree is review-only. Claude may suggest changes but must not edit source files.\"}}"
        exit 0
        ;;
esac

if [[ "$TOOL_NAME" == "Bash" ]]; then
    if echo "$COMMAND" | grep -Eq '(^|[;&|[:space:]])(rm|mv|cp|sed -i|perl -pi|tee)([[:space:]]|$)|>|>>'; then
        echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Potentially file-modifying shell command blocked. Ask the user to run or approve manually.\"}}"
        exit 0
    fi
fi

exit 0
