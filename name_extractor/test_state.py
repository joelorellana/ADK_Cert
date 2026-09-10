"""Test session-state access directly.

Run with: python name_extractor/test_state.py
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from agent import root_agent


APP_NAME = "name_extractor_app"
USER_ID = "test_user"
SESSION_ID = "test_session"

load_dotenv(Path(__file__).with_name(".env"))


async def run_turn(runner: Runner, message: str) -> str | None:
    content = Content(
        role="user",
        parts=[Part.from_text(text=message)],
    )

    final_response = None
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text

    return final_response


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
        print("=== Running agent ===")
        response = await run_turn(runner, "Hi, my name is Alex Johnson")
        print(f"Agent response: {response}")

        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        if session is None:
            raise RuntimeError("Session was not found after agent execution")

        print("\n=== State after execution ===")
        print(f"Full state: {session.state}")
        print(f"Extracted name: {session.state.get('user_name')}")

        if session.state.get("user_name"):
            print("Name was successfully extracted and stored.")
        else:
            print("Name extraction failed.")

        print("\n=== Simulating second turn ===")
        response = await run_turn(runner, "What's my name?")
        print(f"Agent response: {response}")

        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        if session is None:
            raise RuntimeError("Session was not found after the second turn")

        print(f"State still contains: {session.state.get('user_name')}")
        print("State persists across turns.")
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
