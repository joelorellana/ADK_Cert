"""Namespace Demo - Shows all four state namespaces.

Demonstrates temp, session, user, and app persistence scopes with ADK 2.8.

Reference: https://google.github.io/adk-docs/sessions/state.md
"""

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext


def build_instruction(context: ReadonlyContext) -> str:
    """Build instructions from all four state namespaces."""
    app_name = context.state.get("app:name") or "Namespace Demo"
    app_version = context.state.get("app:version") or "1.0"
    user_theme = context.state.get("user:theme") or "not set"
    topic = context.state.get("topic") or "not set"
    current_step = context.state.get("temp:step") or "not set"

    return f"""\
You are a demo assistant showing state namespaces.

=== App State (global for all users) ===
App name: {app_name}
App version: {app_version}

=== User State (persists across sessions) ===
User preference: {user_theme}

=== Session State (persists this conversation) ===
Conversation topic: {topic}

=== Temp State (current turn only) ===
Current step: {current_step}

Respond with a friendly message showing these namespace values.
"""


root_agent = LlmAgent(
    model="gemini-3.5-flash",
    name="namespace_demo",
    instruction=build_instruction,
    output_key="response",
)
