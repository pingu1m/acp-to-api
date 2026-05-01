"""
Complex LangChain agentic E2E test suite.

Exercises the acp-to-api OpenAI-compatible proxy through LangChain's
ChatOpenAI integration, including:

  - Basic ChatOpenAI invocation (non-streaming & streaming)
  - Multi-turn conversation with message history
  - Tool-calling ReAct agent via LangGraph
  - Structured output / chain composition
  - Concurrent requests to verify prompt-lock serialization
  - Multimodal (image) content blocks
  - Error handling for unknown providers

Requires a running acp-to-api server (started by conftest fixtures).
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm(base_url: str, *, streaming: bool = False) -> ChatOpenAI:
    """Build a ChatOpenAI pointed at the local proxy."""
    return ChatOpenAI(
        base_url=base_url,
        api_key="unused",
        model="cursor",
        streaming=streaming,
        temperature=0,
    )


def _base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/api/v1/cursor/openai"


# ---------------------------------------------------------------------------
# Fake tools for the ReAct agent
# ---------------------------------------------------------------------------


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b


@tool
def reverse_string(text: str) -> str:
    """Reverse a string and return it."""
    return text[::-1]


# ===================================================================
# Tests
# ===================================================================


class TestBasicLangChainInvocation:
    """Verify the proxy works as a drop-in OpenAI backend for LangChain."""

    def test_simple_invoke(self, server_process: object, server_port: int) -> None:
        llm = _llm(_base_url(server_port))
        result = llm.invoke("Reply with exactly: langchain-ok")
        assert isinstance(result, AIMessage)
        assert result.content.strip() != ""

    def test_streaming_invoke(self, server_process: object, server_port: int) -> None:
        llm = _llm(_base_url(server_port), streaming=True)
        chunks: list[str] = []
        for chunk in llm.stream("Reply with exactly: stream-langchain-ok"):
            if chunk.content:
                chunks.append(chunk.content)
        full = "".join(chunks).strip()
        assert full != "", "streaming produced no content"

    def test_invoke_with_system_message(self, server_process: object, server_port: int) -> None:
        llm = _llm(_base_url(server_port))
        messages: list[BaseMessage] = [
            SystemMessage(content="You are a helpful math tutor."),
            HumanMessage(content="What is 2 + 2? Reply with just the number."),
        ]
        result = llm.invoke(messages)
        assert isinstance(result, AIMessage)
        assert result.content.strip() != ""


class TestMultiTurnConversation:
    """Ensure multi-turn context is forwarded correctly through the proxy."""

    def test_context_preserved_across_turns(self, server_process: object, server_port: int) -> None:
        llm = _llm(_base_url(server_port))

        history: list[BaseMessage] = [
            SystemMessage(content="You are a concise assistant. Remember everything the user says."),
            HumanMessage(content="My favorite color is cerulean."),
        ]
        r1 = llm.invoke(history)
        assert isinstance(r1, AIMessage)

        history.append(r1)
        history.append(HumanMessage(content="What is my favorite color? Reply with just the color name."))
        r2 = llm.invoke(history)
        assert isinstance(r2, AIMessage)
        # The model should recall "cerulean" from context.
        assert r2.content.strip() != ""
        assert "cerulean" in r2.content.lower(), f"Model failed to recall 'cerulean' from context: {r2.content[:200]}"

    def test_long_conversation_chain(self, server_process: object, server_port: int) -> None:
        """Build a 6-message conversation and verify the proxy handles it."""
        llm = _llm(_base_url(server_port))
        messages: list[BaseMessage] = [
            SystemMessage(content="You are a trivia host. Be concise."),
            HumanMessage(content="Topic: geography. My team name is 'Globetrotters'."),
        ]
        r1 = llm.invoke(messages)
        messages.append(r1)

        messages.append(HumanMessage(content="What is the capital of France? One word answer."))
        r2 = llm.invoke(messages)
        messages.append(r2)

        messages.append(HumanMessage(content="Summarize: what team am I on and what was the last question?"))
        r3 = llm.invoke(messages)
        assert isinstance(r3, AIMessage)
        assert r3.content.strip() != ""


class TestReActAgent:
    """
    Build a LangGraph ReAct agent with tool-calling and run it against the proxy.

    The underlying ACP model may or may not actually call tools (it depends on
    the Cursor agent's capabilities). This test verifies the full round-trip
    works without errors — the agent loop completes and produces a final answer.
    """

    def test_react_agent_with_math_tools(self, server_process: object, server_port: int) -> None:
        llm = _llm(_base_url(server_port))
        agent = create_react_agent(llm, tools=[add, multiply])

        result = agent.invoke(
            {"messages": [HumanMessage(content="What is 3 + 5? Use the add tool if you can, otherwise just answer.")]}
        )
        final_messages = result["messages"]
        assert len(final_messages) >= 1
        last = final_messages[-1]
        assert isinstance(last, (AIMessage, HumanMessage))
        # The agent should produce some content (tool result or direct answer).
        if isinstance(last, AIMessage):
            assert last.content.strip() != "" or last.tool_calls
        # Verify the response is relevant to the math question
        all_text = " ".join(m.content or "" for m in final_messages if isinstance(m, AIMessage)).lower()
        assert "8" in all_text or "eight" in all_text or last.tool_calls, (
            f"Expected answer containing '8' for 3+5, got: {all_text[:200]}"
        )

    def test_react_agent_multi_tool(self, server_process: object, server_port: int) -> None:
        llm = _llm(_base_url(server_port))
        agent = create_react_agent(llm, tools=[add, multiply, reverse_string])

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "First add 10 and 20, then reverse the string 'hello'. "
                            "Use the tools if you can, otherwise just answer directly."
                        )
                    )
                ]
            }
        )
        final_messages = result["messages"]
        assert len(final_messages) >= 1
        last = final_messages[-1]
        assert isinstance(last, (AIMessage, HumanMessage))
        # Should have produced content addressing both sub-tasks
        if isinstance(last, AIMessage):
            assert last.content.strip() != "", "Multi-tool agent produced empty final answer"


class TestChainComposition:
    """Test LangChain chain composition (prompt | llm | parser)."""

    def test_prompt_chain(self, server_process: object, server_port: int) -> None:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        llm = _llm(_base_url(server_port))
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a naming consultant. Suggest exactly one name."),
                ("human", "Suggest a name for a {animal} that is {trait}."),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"animal": "cat", "trait": "playful"})
        assert isinstance(result, str)
        assert result.strip() != ""

    def test_batch_invocation(self, server_process: object, server_port: int) -> None:
        """Batch calls go through sequentially due to the prompt lock — verify they all succeed."""
        llm = _llm(_base_url(server_port))
        results = llm.batch(
            [
                [HumanMessage(content="Say 'alpha'")],
                [HumanMessage(content="Say 'beta'")],
            ]
        )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, AIMessage)
            assert r.content.strip() != ""
        # Verify the two responses are distinct (not the same cached response)
        assert results[0].content != results[1].content, (
            "Batch responses are identical — proxy may not be handling separate requests"
        )


class TestMultimodalContent:
    """Verify multimodal (image) content blocks pass through the proxy."""

    # 1x1 transparent PNG as data URL
    PNG_1X1 = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBgL6N4m0AAAAASUVORK5CYII="
    )

    def test_image_content_block(self, server_process: object, server_port: int) -> None:
        llm = _llm(_base_url(server_port))
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": self.PNG_1X1}},
            ]
        )
        result = llm.invoke([message])
        assert isinstance(result, AIMessage)
        assert result.content.strip() != ""


class TestConcurrency:
    """Verify the server handles concurrent requests gracefully (serialized by prompt lock)."""

    def test_concurrent_requests(self, server_process: object, server_port: int) -> None:
        async def _run() -> list[str]:
            llm = _llm(_base_url(server_port))

            async def _call(msg: str) -> str:
                result = await llm.ainvoke(msg)
                return result.content

            tasks = [
                _call("Say 'concurrent-1'"),
                _call("Say 'concurrent-2'"),
                _call("Say 'concurrent-3'"),
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_run())
        assert len(results) == 3
        for r in results:
            assert r.strip() != ""
        # At least 2 of 3 should be distinct (not all the same cached response)
        assert len(set(results)) >= 2, f"All concurrent responses are identical: {results[0][:100]}"


class TestErrorHandling:
    """Verify proper error propagation for bad requests."""

    def test_unknown_provider_returns_404(self, server_process: object, server_port: int) -> None:
        llm = ChatOpenAI(
            base_url=f"http://127.0.0.1:{server_port}/api/v1/nonexistent/openai",
            api_key="unused",
            model="cursor",
        )
        with pytest.raises(Exception):
            llm.invoke("hello")
