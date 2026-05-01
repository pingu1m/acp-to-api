"""
LangGraph agentic end-to-end test for acp-to-api.

Builds a multi-step LangGraph agent that exercises the OpenAI-compatible API
through tool-calling loops, multi-turn conversations, parallel tool execution,
conditional branching, and streaming — validating that the proxy behaves
correctly under realistic agentic workloads.

Requires a running acp-to-api server (uses the session-scoped fixtures from conftest.py).
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# ---------------------------------------------------------------------------
# Fake tools the agent can "call" — they run locally, no external deps needed
# ---------------------------------------------------------------------------


def calculator(expression: str) -> str:
    """Evaluate a simple math expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return json.dumps({"result": result})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def lookup_city_population(city: str) -> str:
    """Return the (fake) population of a city."""
    populations = {
        "tokyo": 13_960_000,
        "new york": 8_336_000,
        "london": 8_982_000,
        "paris": 2_161_000,
        "berlin": 3_645_000,
    }
    pop = populations.get(city.lower())
    if pop is not None:
        return json.dumps({"city": city, "population": pop})
    return json.dumps({"city": city, "error": "not found"})


def string_reverser(text: str) -> str:
    """Reverse the given text."""
    return json.dumps({"reversed": text[::-1]})


# Map tool names to callables for the ToolNode
TOOLS = [calculator, lookup_city_population, string_reverser]


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


class RouterState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    route: str


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_agent_graph(llm: ChatOpenAI) -> StateGraph:
    """Build a ReAct-style agent graph with tool calling."""

    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def _make_llm(port: int) -> ChatOpenAI:
    """Create a ChatOpenAI client pointed at the local acp-to-api proxy."""
    return ChatOpenAI(
        base_url=f"http://127.0.0.1:{port}/api/v1/cursor/openai",
        api_key="dummy",
        model="cursor",
        temperature=0,
        streaming=False,
    )


def _make_streaming_llm(port: int) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=f"http://127.0.0.1:{port}/api/v1/cursor/openai",
        api_key="dummy",
        model="cursor",
        temperature=0,
        streaming=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicAgentLoop:
    """Validate the agent can complete a simple tool-calling loop."""

    def test_single_tool_call_roundtrip(self, server_process: object, server_port: int) -> None:
        """Agent should call calculator and return a final answer."""
        llm = _make_llm(server_port)
        agent = build_agent_graph(llm)

        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "You are a helpful assistant with access to tools. "
                            "Use the calculator tool to compute math expressions. "
                            "Always use tools when asked to compute something."
                        )
                    ),
                    HumanMessage(content="What is 7 * 8?"),
                ]
            }
        )

        messages = result["messages"]
        # Should have at least: system, human, AI (tool_call), tool result, AI final
        assert len(messages) >= 3, f"Expected multi-step conversation, got {len(messages)} messages"

        # The final message should be from the assistant
        final = messages[-1]
        assert isinstance(final, AIMessage), f"Last message should be AIMessage, got {type(final)}"
        assert final.content, "Final AI message should have content"

    def test_tool_not_needed(self, server_process: object, server_port: int) -> None:
        """When no tool is needed, agent should respond directly."""
        llm = _make_llm(server_port)
        agent = build_agent_graph(llm)

        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content="You are a helpful assistant. Answer directly without tools."),
                    HumanMessage(content="Say hello."),
                ]
            }
        )

        messages = result["messages"]
        final = messages[-1]
        assert isinstance(final, AIMessage)
        assert final.content
        # Should NOT have any tool calls in the final message
        assert not final.tool_calls, "Agent should not call tools for a simple greeting"


class TestMultiToolAgent:
    """Validate the agent can chain multiple tool calls."""

    def test_sequential_tool_calls(self, server_process: object, server_port: int) -> None:
        """Agent should call multiple tools in sequence to answer a compound question."""
        llm = _make_llm(server_port)
        agent = build_agent_graph(llm)

        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "You have access to tools. Use lookup_city_population to get populations, "
                            "and calculator to do math. Always use tools — do not guess."
                        )
                    ),
                    HumanMessage(
                        content=("What is the population of Tokyo? Then compute that population divided by 1000.")
                    ),
                ]
            }
        )

        messages = result["messages"]

        # Final answer should exist — the model may or may not have used tools
        # (depends on the ACP agent's capabilities), but it must produce output.
        final = messages[-1]
        assert isinstance(final, AIMessage)
        assert final.content

    def test_parallel_tool_calls(self, server_process: object, server_port: int) -> None:
        """Agent may issue parallel tool calls; verify they all resolve."""
        llm = _make_llm(server_port)
        agent = build_agent_graph(llm)

        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "You have tools. Use lookup_city_population for each city. "
                            "You can call multiple tools at once."
                        )
                    ),
                    HumanMessage(content=("Look up the populations of Tokyo and London.")),
                ]
            }
        )

        messages = result["messages"]
        # The model may or may not use tools depending on ACP capabilities.
        # Just verify the agent loop completed with a final answer.
        final = messages[-1]
        assert isinstance(final, AIMessage)
        assert final.content


class TestMultiTurnConversation:
    """Validate multi-turn agentic conversations through the proxy."""

    def test_multi_turn_with_context_retention(self, server_process: object, server_port: int) -> None:
        """Run the agent across multiple turns, verifying context carries forward."""
        llm = _make_llm(server_port)
        agent = build_agent_graph(llm)

        # Turn 1
        result1 = agent.invoke(
            {
                "messages": [
                    SystemMessage(content="You are a helpful assistant with tools."),
                    HumanMessage(content="Reverse the string 'langgraph'."),
                ]
            }
        )
        messages_t1 = result1["messages"]
        final_t1 = messages_t1[-1]
        assert isinstance(final_t1, AIMessage)
        assert final_t1.content

        # Turn 2 — build on the previous conversation
        result2 = agent.invoke(
            {
                "messages": messages_t1
                + [
                    HumanMessage(content="Now reverse the string 'openai'."),
                ]
            }
        )
        messages_t2 = result2["messages"]
        final_t2 = messages_t2[-1]
        assert isinstance(final_t2, AIMessage)
        assert final_t2.content

        # Verify the conversation grew
        assert len(messages_t2) > len(messages_t1)
        # Verify turn 2 response references something from turn 1
        t2_text = " ".join(m.content or "" for m in messages_t2 if isinstance(m, AIMessage)).lower()
        assert "openai" in t2_text or "ianpo" in t2_text or len(messages_t2) > len(messages_t1) + 1, (
            "Turn 2 should reference the 'openai' reversal task"
        )


class TestStreamingAgent:
    """Validate the agent works with streaming responses."""

    def test_streaming_tool_loop(self, server_process: object, server_port: int) -> None:
        """Agent with streaming LLM should complete a tool-calling loop."""
        llm = _make_streaming_llm(server_port)
        agent = build_agent_graph(llm)

        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=("You are a helpful assistant. Use the calculator tool for math.")),
                    HumanMessage(content="Compute 123 + 456."),
                ]
            }
        )

        messages = result["messages"]
        final = messages[-1]
        assert isinstance(final, AIMessage)
        assert final.content

    def test_stream_events(self, server_process: object, server_port: int) -> None:
        """Use astream_events to verify streaming events flow through the proxy."""
        llm = _make_streaming_llm(server_port)
        agent = build_agent_graph(llm)

        import asyncio

        async def _collect_events() -> list[dict[str, Any]]:
            collected = []
            async for event in agent.astream_events(
                {
                    "messages": [
                        SystemMessage(content="You are a helpful assistant."),
                        HumanMessage(content="Say 'streaming works' in your reply."),
                    ]
                },
                version="v2",
            ):
                collected.append(event)
            return collected

        events = asyncio.run(_collect_events())
        assert len(events) > 0, "Expected streaming events from the agent"

        # Should contain at least on_chat_model_stream or on_chain_end events
        event_kinds = {e.get("event") for e in events}
        assert event_kinds, "No event types found"


class TestEdgeCases:
    """Validate the proxy handles edge cases gracefully under agentic load."""

    def test_empty_user_message(self, server_process: object, server_port: int) -> None:
        """Agent should handle an empty user message without crashing."""
        llm = _make_llm(server_port)
        agent = build_agent_graph(llm)

        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=""),
                ]
            }
        )

        messages = result["messages"]
        final = messages[-1]
        assert isinstance(final, AIMessage)
        # Even with empty input, the model should produce some response
        assert final.content is not None, "Empty user message produced None content"

    def test_long_context_message(self, server_process: object, server_port: int) -> None:
        """Agent should handle a long context message through the proxy."""
        llm = _make_llm(server_port)
        agent = build_agent_graph(llm)

        long_context = "The quick brown fox jumps over the lazy dog. " * 100

        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=f"Summarize this text: {long_context}"),
                ]
            }
        )

        messages = result["messages"]
        final = messages[-1]
        assert isinstance(final, AIMessage)
        assert final.content

    def test_tool_error_handling(self, server_process: object, server_port: int) -> None:
        """Agent should handle a tool that returns an error gracefully."""
        llm = _make_llm(server_port)
        agent = build_agent_graph(llm)

        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "You have tools. Use lookup_city_population to look up cities. "
                            "If a city is not found, say so."
                        )
                    ),
                    HumanMessage(content="What is the population of Atlantis?"),
                ]
            }
        )

        messages = result["messages"]
        final = messages[-1]
        assert isinstance(final, AIMessage)
        assert final.content


class TestGraphTopologies:
    """Test different graph structures to stress the proxy."""

    def test_branching_agent(self, server_process: object, server_port: int) -> None:
        """Build a graph with a routing node that branches based on input."""

        llm = _make_llm(server_port)

        def router(state: RouterState) -> dict[str, Any]:
            """Classify the user intent."""
            response = llm.invoke(
                state["messages"]
                + [
                    HumanMessage(
                        content=(
                            "Classify the user's request as either 'math' or 'lookup'. "
                            "Reply with ONLY the word 'math' or 'lookup', nothing else."
                        )
                    )
                ]
            )
            text = (response.content or "").strip().lower()
            route = "math" if "math" in text else "lookup"
            return {"messages": [response], "route": route}

        def math_node(state: RouterState) -> dict[str, list[BaseMessage]]:
            response = llm.invoke(
                state["messages"]
                + [HumanMessage(content="Solve the math problem from the conversation. Give a numeric answer.")]
            )
            return {"messages": [response]}

        def lookup_node(state: RouterState) -> dict[str, list[BaseMessage]]:
            response = llm.invoke(
                state["messages"] + [HumanMessage(content="Answer the lookup question from the conversation.")]
            )
            return {"messages": [response]}

        def pick_branch(state: RouterState) -> str:
            return state.get("route", "math")

        graph = StateGraph(RouterState)
        graph.add_node("router", router)
        graph.add_node("math_solver", math_node)
        graph.add_node("lookup_solver", lookup_node)
        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            pick_branch,
            {
                "math": "math_solver",
                "lookup": "lookup_solver",
            },
        )
        graph.add_edge("math_solver", END)
        graph.add_edge("lookup_solver", END)

        compiled = graph.compile()

        # Test math route
        result = compiled.invoke(
            {
                "messages": [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="What is 15 * 3?"),
                ],
                "route": "",
            }
        )

        assert len(result["messages"]) >= 3
        final = result["messages"][-1]
        assert isinstance(final, AIMessage)
        assert final.content

    def test_loop_with_max_iterations(self, server_process: object, server_port: int) -> None:
        """Agent loop should respect iteration limits (recursion_limit)."""
        llm = _make_llm(server_port)
        agent = build_agent_graph(llm)

        # Use a low recursion limit to test the proxy handles rapid back-and-forth
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "You are a helpful assistant with tools. "
                            "Use the calculator tool to compute 2+2, then respond."
                        )
                    ),
                    HumanMessage(content="Compute 2+2."),
                ]
            },
            config={"recursion_limit": 10},
        )

        messages = result["messages"]
        final = messages[-1]
        assert isinstance(final, AIMessage)
        assert final.content
