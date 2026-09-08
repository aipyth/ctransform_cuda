#!/usr/bin/env bash
set -euo pipefail

INPUT="$(cat)"
TOOL_NAME="$(echo "$INPUT" | jq -r '.tool_name // empty')"
FILE_PATH="$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')"
COMMAND="$(echo "$INPUT" | jq -r '.tool_input.command // empty')"

# case "$FILE_PATH" in
    # */src/*|*/include/*|*/cuda/*|*/tests/*|*/python/*)

case "$FILE_PATH" in
    (*/src/*|*/include/*|*/cuda/*|*/tests/*|*/python/*)
        echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Source tree is review-only. Claude may suggest changes but must not edit source files.\"}}"
        exit 0
        ;;
esac

if [[ "$TOOL_NAME" == "Bash" ]]; then
    # Strip redirections that cannot modify a file (2>/dev/null, 2>&1, &>/dev/null,
    # >&2, ...) so the redirection check below only sees real writes.
    SANITIZED="$(printf '%s' "$COMMAND" \
        | sed -E 's#[0-9]*(>>?|&>)[[:space:]]*(&[0-9-]+|/dev/(null|stdout|stderr))##g')"

    # Destructive commands, as whole words.
    DESTRUCTIVE='(^|[;&|[:space:]])(rm|mv|cp|tee|truncate|dd|shred)([[:space:]]|$)'
    # In-place editors: sed/perl with an -i flag, possibly after other short flags.
    INPLACE='(^|[;&|[:space:]])(sed|perl)([[:space:]]+-[a-zA-Z]+)*[[:space:]]+-[a-zA-Z]*i'
    # A redirection operator in operator position (preceded by start-of-line or
    # whitespace), so `grep 'a>b'` and `-->` no longer trip the guard.
    REDIRECT='(^|[[:space:]])[0-9]*&?>>?[[:space:]]*[^[:space:];&|]'

    if printf '%s' "$SANITIZED" | grep -Eq "$DESTRUCTIVE|$INPLACE|$REDIRECT"; then
        echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Potentially file-modifying shell command blocked. Ask the user to run or approve manually.\"}}"
        exit 0
    fi
fi

exit 0
