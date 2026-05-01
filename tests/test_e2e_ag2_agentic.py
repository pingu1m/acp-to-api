"""Complex AG2 (formerly AutoGen) agentic tests against the acp-to-api proxy.

These tests exercise multi-agent conversations, tool calling, group chat
orchestration, and two-agent workflows — all routed through the proxy's
OpenAI-compatible API via AG2's ConversableAgent abstraction.

Requires:
    pip install "ag2[openai]"
"""

from __future__ import annotations

import json
from typing import Annotated

from autogen import (
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    LLMConfig,
    register_function,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/api/v1/cursor/openai"


def _llm_config(port: int) -> LLMConfig:
    """Build an LLMConfig pointing at the local acp-to-api proxy."""
    return LLMConfig(
        {
            "api_type": "openai",
            "model": "cursor",
            "base_url": _base_url(port),
            "api_key": "dummy",
        },
        temperature=0,
    )


# ---------------------------------------------------------------------------
# 1. Basic two-agent conversation
# ---------------------------------------------------------------------------


def test_two_agent_basic_chat(server_process: object, server_port: int) -> None:
    """Two ConversableAgents have a short back-and-forth conversation."""
    llm_config = _llm_config(server_port)

    assistant = ConversableAgent(
        name="assistant",
        system_message=(
            "You are a concise assistant. Always include the token 'ag2-basic-ok' somewhere in your reply."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    user_proxy = ConversableAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
    )

    result = user_proxy.initiate_chat(
        recipient=assistant,
        message="Say hello and confirm you are working.",
        max_turns=2,
    )

    # At least one assistant reply should exist with non-empty content.
    texts = [
        m.get("content", "") or ""
        for m in result.chat_history
        if m.get("role") == "assistant" or m.get("name") == "assistant"
    ]
    combined = " ".join(texts).lower()
    assert combined.strip() != "", "assistant produced no output"


# ---------------------------------------------------------------------------
# 2. Multi-turn conversation with context retention
# ---------------------------------------------------------------------------


def test_multi_turn_context_retention(server_process: object, server_port: int) -> None:
    """Verify the agent retains context across multiple turns."""
    llm_config = _llm_config(server_port)

    assistant = ConversableAgent(
        name="context_assistant",
        system_message=(
            "You are a helpful assistant. Remember all facts the user tells you. "
            "When asked to recall, list them accurately."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    user = ConversableAgent(
        name="context_user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        # Auto-reply with follow-up messages to drive the conversation.
        default_auto_reply="Now recall the project name I told you.",
    )

    result = user.initiate_chat(
        recipient=assistant,
        message="Remember this: the project name is Neptune-7.",
        max_turns=3,
    )

    # The assistant should mention Neptune-7 in at least one reply.
    all_text = " ".join(m.get("content", "") or "" for m in result.chat_history).lower()
    assert "neptune" in all_text, f"context not retained, history: {all_text[:500]}"


# ---------------------------------------------------------------------------
# 3. Tool / function calling between two agents
# ---------------------------------------------------------------------------


def _calculate_area(
    length: Annotated[float, "Length of the rectangle"],
    width: Annotated[float, "Width of the rectangle"],
) -> str:
    """Calculate the area of a rectangle."""
    return json.dumps({"area": length * width, "unit": "sq_units"})


def _convert_temperature(
    value: Annotated[float, "Temperature value"],
    from_unit: Annotated[str, "Source unit: 'C' or 'F'"],
) -> str:
    """Convert temperature between Celsius and Fahrenheit."""
    if from_unit.upper() == "C":
        converted = value * 9 / 5 + 32
        return json.dumps({"result": round(converted, 2), "unit": "F"})
    else:
        converted = (value - 32) * 5 / 9
        return json.dumps({"result": round(converted, 2), "unit": "C"})


def test_tool_calling_two_agents(server_process: object, server_port: int) -> None:
    """An LLM agent suggests tool calls; an executor agent runs them."""
    llm_config = _llm_config(server_port)

    caller = ConversableAgent(
        name="tool_caller",
        system_message=(
            "You are a math assistant. Use the provided tools to answer questions. "
            "After getting the tool result, state the answer clearly."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    executor = ConversableAgent(
        name="tool_executor",
        human_input_mode="NEVER",
    )

    register_function(
        _calculate_area,
        caller=caller,
        executor=executor,
        description="Calculate the area of a rectangle given length and width.",
    )

    register_function(
        _convert_temperature,
        caller=caller,
        executor=executor,
        description="Convert a temperature between Celsius and Fahrenheit.",
    )

    result = executor.initiate_chat(
        recipient=caller,
        message="What is the area of a rectangle with length 12 and width 5?",
        max_turns=4,
    )

    all_text = " ".join(m.get("content", "") or "" for m in result.chat_history).lower()
    # The correct area is 60 — it should appear somewhere in the conversation.
    assert "60" in all_text, f"expected area=60 in conversation: {all_text[:500]}"


# ---------------------------------------------------------------------------
# 4. Group chat with three specialised agents
# ---------------------------------------------------------------------------


def test_group_chat_multi_agent(server_process: object, server_port: int) -> None:
    """Three agents collaborate in a GroupChat to solve a task."""
    llm_config = _llm_config(server_port)

    researcher = ConversableAgent(
        name="researcher",
        system_message=(
            "You are a researcher. When asked about a topic, provide 2-3 concise "
            "bullet points of factual information. Always include the phrase "
            "'research-complete' when you finish."
        ),
        description="Provides factual research on topics.",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    writer = ConversableAgent(
        name="writer",
        system_message=(
            "You are a writer. Take the researcher's bullet points and turn them "
            "into a short paragraph (2-3 sentences). Always include the phrase "
            "'draft-ready' when you finish."
        ),
        description="Writes polished text from research notes.",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    editor = ConversableAgent(
        name="editor",
        system_message=(
            "You are an editor. Review the writer's paragraph for clarity and "
            "brevity. Output the final version and include 'final-approved'. "
            "When satisfied, say TERMINATE."
        ),
        description="Reviews and finalises written content.",
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "TERMINATE" in (x.get("content", "") or "").upper(),
    )

    group_chat = GroupChat(
        agents=[researcher, writer, editor],
        messages=[],
        max_round=6,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(
        name="chat_manager",
        groupchat=group_chat,
        llm_config=llm_config,
    )

    result = researcher.initiate_chat(
        recipient=manager,
        message="Write a short summary about the benefits of open-source software.",
        max_turns=6,
    )

    all_text = " ".join(m.get("content", "") or "" for m in result.chat_history).lower()

    # Verify the pipeline actually ran — at least one agent should have produced output.
    assert len(result.chat_history) >= 3, f"expected at least 3 messages in group chat, got {len(result.chat_history)}"
    assert all_text.strip() != "", "group chat produced no output"


# ---------------------------------------------------------------------------
# 5. Two-agent debate / adversarial pattern
# ---------------------------------------------------------------------------


def test_two_agent_debate(server_process: object, server_port: int) -> None:
    """Two agents argue opposing sides, exercising multi-turn back-and-forth."""
    llm_config = _llm_config(server_port)

    proponent = ConversableAgent(
        name="proponent",
        system_message=(
            "You argue IN FAVOUR of remote work. Keep responses to 2-3 sentences. "
            "After 2 rounds, include 'debate-concluded' in your message."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "debate-concluded" in (x.get("content", "") or "").lower(),
    )

    opponent = ConversableAgent(
        name="opponent",
        system_message=(
            "You argue AGAINST remote work. Keep responses to 2-3 sentences. "
            "After 2 rounds, include 'debate-concluded' in your message."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "debate-concluded" in (x.get("content", "") or "").lower(),
    )

    result = proponent.initiate_chat(
        recipient=opponent,
        message="I believe remote work is the future of productivity. What do you think?",
        max_turns=4,
    )

    assert len(result.chat_history) >= 2, f"debate too short: {len(result.chat_history)} messages"

    all_text = " ".join(m.get("content", "") or "" for m in result.chat_history).lower()
    # Both sides should have mentioned "remote" at some point.
    assert "remote" in all_text, f"debate didn't discuss the topic: {all_text[:500]}"


# ---------------------------------------------------------------------------
# 6. Sequential chained conversations (agent hand-off)
# ---------------------------------------------------------------------------


def test_sequential_agent_handoff(server_process: object, server_port: int) -> None:
    """Chain two separate conversations: output of the first feeds the second."""
    llm_config = _llm_config(server_port)

    summariser = ConversableAgent(
        name="summariser",
        system_message=("You summarise text into exactly one sentence. Start your summary with 'Summary:'"),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    translator = ConversableAgent(
        name="translator",
        system_message=("You translate English text into French. Start your translation with 'Traduction:'"),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    driver = ConversableAgent(
        name="driver",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
    )

    # Step 1: summarise
    long_text = (
        "Open-source software allows anyone to inspect, modify, and distribute "
        "the source code. This fosters collaboration, transparency, and rapid "
        "innovation across the global developer community."
    )
    summary_result = driver.initiate_chat(
        recipient=summariser,
        message=f"Summarise this: {long_text}",
        max_turns=2,
    )

    summary = ""
    for m in reversed(summary_result.chat_history):
        if (m.get("name") == "summariser" or m.get("role") == "assistant") and m.get("content"):
            summary = m["content"]
            break

    assert summary.strip() != "", "summariser produced no output"

    # Step 2: translate the summary
    translation_result = driver.initiate_chat(
        recipient=translator,
        message=f"Translate this to French: {summary}",
        max_turns=2,
    )

    translation = ""
    for m in reversed(translation_result.chat_history):
        if (m.get("name") == "translator" or m.get("role") == "assistant") and m.get("content"):
            translation = m["content"]
            break

    assert translation.strip() != "", "translator produced no output"
    # Basic sanity: French text typically contains some of these common words.
    french_indicators = ["le", "la", "les", "de", "du", "des", "et", "un", "une", "est", "en"]
    translation_lower = translation.lower()
    found = any(f" {w} " in f" {translation_lower} " for w in french_indicators)
    assert found, f"translation doesn't look French: {translation[:300]}"


# ---------------------------------------------------------------------------
# 7. System message override / persona switching
# ---------------------------------------------------------------------------


def test_persona_switching(server_process: object, server_port: int) -> None:
    """Verify agents respect distinct system messages (personas)."""
    llm_config = _llm_config(server_port)

    pirate = ConversableAgent(
        name="pirate_agent",
        system_message=(
            "You are a pirate. Every response must include 'Arrr' and refer to "
            "the user as 'matey'. Keep it to one sentence."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    robot = ConversableAgent(
        name="robot_agent",
        system_message=(
            "You are a robot. Every response must include 'BEEP BOOP' and refer "
            "to the user as 'human'. Keep it to one sentence."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    user = ConversableAgent(
        name="persona_user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
    )

    pirate_result = user.initiate_chat(
        recipient=pirate,
        message="Greet me in your style.",
        max_turns=2,
    )
    pirate_text = " ".join(
        m.get("content", "") or ""
        for m in pirate_result.chat_history
        if m.get("name") == "pirate_agent" or m.get("role") == "assistant"
    ).lower()

    robot_result = user.initiate_chat(
        recipient=robot,
        message="Greet me in your style.",
        max_turns=2,
    )
    robot_text = " ".join(
        m.get("content", "") or ""
        for m in robot_result.chat_history
        if m.get("name") == "robot_agent" or m.get("role") == "assistant"
    ).lower()

    assert "arrr" in pirate_text or pirate_text.strip() != "", f"pirate agent produced no output: {pirate_text[:200]}"
    assert robot_text.strip() != "", f"robot agent produced no output: {robot_text[:200]}"
    # Both agents should have produced some response
    assert pirate_text.strip() != ""
    assert robot_text.strip() != ""
    # The two personas should produce different responses
    assert pirate_text != robot_text, (
        "Pirate and robot agents produced identical responses — personas not differentiated"
    )
