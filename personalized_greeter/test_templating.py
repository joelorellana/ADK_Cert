"""Test dynamic instructions with different session-state values.

Run with: python test_templating.py
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from agent import root_agent


APP_NAME = "greeter_app"
USER_ID = "user1"
SESSION_ID = "session1"

load_dotenv(Path(__file__).with_name(".env"))


async def run_prompt(
    runner: Runner,
    text: str,
    state_delta: dict[str, str] | None = None,
) -> None:
    message = Content(
        role="user",
        parts=[Part.from_text(text=text)],
    )

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message,
        state_delta=state_delta,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(f"Agent: {event.content.parts[0].text}\n")


async def main() -> None:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    try:
        print("=== Test 1: No state (all defaults) ===")
        await run_prompt(runner, "Hello")

        print("=== Test 2: With user name ===")
        await run_prompt(
            runner,
            "Hello again",
            state_delta={"user_name": "Alex"},
        )

        print("=== Test 3: With all state values ===")
        await run_prompt(
            runner,
            "Hola de nuevo",
            state_delta={
                "user_language": "Spanish",
                "membership_tier": "premium",
            },
        )

        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        if session is None:
            raise RuntimeError("Session was not found after agent execution")

        print("=== Current state ===")
        print(session.state)
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
