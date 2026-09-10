"""Test state namespaces and their persistence scopes.

Run with: python test_namespaces.py
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part

from agent import root_agent


APP_NAME = "namespace_demo_app"
USER_ID = "user1"
SESSION_ID = "session1"

load_dotenv(Path(__file__).with_name(".env"))


async def run_prompt(
    runner: Runner,
    session_id: str,
    text: str,
    state_delta: dict[str, str] | None = None,
) -> str | None:
    final_response = None
    message = Content(
        role="user",
        parts=[Part.from_text(text=text)],
    )

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=message,
        state_delta=state_delta,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text

    return final_response


async def get_session(
    session_service: InMemorySessionService,
    session_id: str,
) -> Session:
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    if session is None:
        raise RuntimeError(f"Session not found: {session_id}")
    return session


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

    initial_state = {
        "app:name": "Namespace Demo",
        "app:version": "2.0",
        "user:theme": "dark",
        "topic": "state management",
        "temp:step": "initialization",
    }

    try:
        print("=== Setting state in all namespaces ===")
        print("State before run:", initial_state)

        print("\n=== Running agent (Turn 1) ===")
        response = await run_prompt(
            runner,
            SESSION_ID,
            "Show me the namespace values",
            state_delta=initial_state,
        )
        print(f"Agent response: {response}")

        session = await get_session(session_service, SESSION_ID)

        print("\n=== State after Turn 1 ===")
        print(f"Full state: {session.state}")
        print(f"temp:step: {session.state.get('temp:step')}")
        print(f"topic: {session.state.get('topic')}")
        print(f"user:theme: {session.state.get('user:theme')}")
        print(f"app:version: {session.state.get('app:version')}")

        assert session.state.get("temp:step") is None
        assert session.state.get("topic") == "state management"
        assert session.state.get("user:theme") == "dark"
        assert session.state.get("app:version") == "2.0"

        print("\n=== Simulating Turn 2 (same session) ===")
        response = await run_prompt(
            runner,
            SESSION_ID,
            "Check state again",
        )
        print(f"Agent response: {response}")

        session = await get_session(session_service, SESSION_ID)

        print("\n=== State after Turn 2 ===")
        print(f"temp:step: {session.state.get('temp:step')}")
        print(f"topic: {session.state.get('topic')}")
        print(f"user:theme: {session.state.get('user:theme')}")

        print("\n=== Simulating new session (session2) ===")
        session2 = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id="session2",
        )

        print(f"New session state: {session2.state}")
        print(f"topic: {session2.state.get('topic')}")
        print(f"user:theme: {session2.state.get('user:theme')}")
        print(f"app:version: {session2.state.get('app:version')}")

        assert session2.state.get("topic") is None
        assert session2.state.get("user:theme") == "dark"
        assert session2.state.get("app:version") == "2.0"
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
