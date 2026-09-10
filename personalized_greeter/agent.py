"""Personalized Greeter - Demonstrates dynamic instructions from session state.

Uses an InstructionProvider to apply defaults and conditional content with ADK 2.8.

Reference: https://google.github.io/adk-docs/sessions/state.md
"""

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext


def build_instruction(context: ReadonlyContext) -> str:
    """Build a personalized instruction from the current session state."""
    user_name = context.state.get("user_name") or "there"
    user_language = context.state.get("user_language") or "English"
    membership = context.state.get("membership_tier") or "free"
    membership_details = (
        f"Your membership level is: {membership}."
        if context.state.get("membership_tier")
        else ""
    )

    return f"""\
You are a friendly assistant.

User information:
- Name: {user_name}
- Preferred language: {user_language}
- Membership: {membership}

{membership_details}

Greet the user warmly and offer assistance.
Respond in {user_language}.
Only mention the membership level when membership details are present.
"""


root_agent = LlmAgent(
    model="gemini-3.5-flash",
    name="personalized_greeter",
    instruction=build_instruction,
)
