"""Tests for AIGenerator and Claude API tool calling."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class MockContentBlock:
    """Mock for Anthropic content blocks."""

    def __init__(
        self, block_type, text=None, tool_name=None, tool_input=None, tool_id=None
    ):
        self.type = block_type
        self.text = text
        self.name = tool_name
        self.input = tool_input or {}
        self.id = tool_id


class MockResponse:
    """Mock for Anthropic API response."""

    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class TestAIGeneratorBasic:
    """Basic tests for AIGenerator initialization and direct responses."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_init_creates_client(self, mock_anthropic):
        """Verify AIGenerator initializes Anthropic client."""
        # Act
        generator = AIGenerator(api_key="test-key", model="claude-test")

        # Assert
        mock_anthropic.assert_called_once_with(api_key="test-key")
        assert generator.model == "claude-test"

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_without_tools(self, mock_anthropic):
        """Verify direct response when no tools are provided."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        mock_response = MockResponse(
            content=[MockContentBlock("text", text="This is a direct answer.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-test")

        # Act
        result = generator.generate_response(query="What is Python?")

        # Assert
        assert result == "This is a direct answer."
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Verify conversation history is included in system prompt."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        mock_response = MockResponse(
            content=[MockContentBlock("text", text="Response with context.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-test")

        # Act
        result = generator.generate_response(
            query="Follow up question?",
            conversation_history="User: Previous question\nAssistant: Previous answer",
        )

        # Assert
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Previous conversation" in call_kwargs["system"]
        assert "Previous question" in call_kwargs["system"]


class TestAIGeneratorToolCalling:
    """Tests for AIGenerator tool calling behavior."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_calls_tool_when_needed(self, mock_anthropic):
        """Mock Claude API to return tool_use, verify tool execution."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # First response: Claude wants to use a tool
        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "machine learning"},
                    tool_id="tool_123",
                )
            ],
            stop_reason="tool_use",
        )

        # Second response: Claude's final answer after getting tool results
        final_response = MockResponse(
            content=[MockContentBlock("text", text="Based on the search, ML is...")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Search results: ML basics content"
        )

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Act
        result = generator.generate_response(
            query="Tell me about machine learning",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert result == "Based on the search, ML is..."
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="machine learning"
        )
        # Should have made 2 API calls
        assert mock_client.messages.create.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_results_passed_to_follow_up(self, mock_anthropic):
        """Verify tool results are included in follow-up API call."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "python"},
                    tool_id="tool_456",
                )
            ],
            stop_reason="tool_use",
        )

        final_response = MockResponse(
            content=[MockContentBlock("text", text="Python answer")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python course content here"

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Act
        generator.generate_response(
            query="Explain Python", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert - Check the second API call contains tool results
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Should have: user message, assistant tool_use, user tool_result
        assert len(messages) == 3

        # Last message should be tool result
        tool_result_message = messages[2]
        assert tool_result_message["role"] == "user"
        assert tool_result_message["content"][0]["type"] == "tool_result"
        assert tool_result_message["content"][0]["tool_use_id"] == "tool_456"
        assert (
            tool_result_message["content"][0]["content"] == "Python course content here"
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_tools_included_in_first_request(self, mock_anthropic):
        """Verify tools are passed to initial API call."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        mock_response = MockResponse(
            content=[MockContentBlock("text", text="Direct answer")],
            stop_reason="end_turn",
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-test")

        tools = [
            {"name": "search_course_content", "description": "Search"},
            {"name": "get_course_outline", "description": "Get outline"},
        ]

        # Act
        generator.generate_response(query="Question", tools=tools, tool_manager=Mock())

        # Assert
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_not_executed_without_manager(self, mock_anthropic):
        """Verify tool execution is skipped if no tool_manager provided."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Claude wants to use a tool but no tool_manager provided
        tool_use_response = MockResponse(
            content=[
                MockContentBlock("text", text="I'll search for that."),
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="tool_789",
                ),
            ],
            stop_reason="tool_use",
        )
        mock_client.messages.create.return_value = tool_use_response

        generator = AIGenerator(api_key="test-key", model="claude-test")

        tools = [{"name": "search_course_content", "description": "Search"}]

        # Act - no tool_manager provided
        result = generator.generate_response(
            query="Search for something", tools=tools, tool_manager=None
        )

        # Assert - Should return the text content, not execute tool
        # Note: Current implementation returns first content block's text
        assert result == "I'll search for that."
        assert mock_client.messages.create.call_count == 1


class TestAIGeneratorToolExecution:
    """Tests for tool execution within the tool loop."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_multiple_tool_calls_in_single_response(self, mock_anthropic):
        """Verify multiple tool calls in single response are all executed."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Response with multiple tool calls
        multi_tool_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "python"},
                    tool_id="tool_1",
                ),
                MockContentBlock(
                    "tool_use",
                    tool_name="get_course_outline",
                    tool_input={"course_name": "Python"},
                    tool_id="tool_2",
                ),
            ],
            stop_reason="tool_use",
        )

        final_response = MockResponse(
            content=[MockContentBlock("text", text="Combined answer")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [multi_tool_response, final_response]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Search result", "Outline result"]

        tools = [
            {"name": "search_course_content", "description": "Search"},
            {"name": "get_course_outline", "description": "Outline"},
        ]

        # Act
        result = generator.generate_response(
            query="Tell me about Python course",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert result == "Combined answer"
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_tools_included_in_follow_up_within_max_rounds(self, mock_anthropic):
        """Verify tools ARE included in follow-up call (before max rounds reached)."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="tool_abc",
                )
            ],
            stop_reason="tool_use",
        )

        final_response = MockResponse(
            content=[MockContentBlock("text", text="Final answer")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        tools = [{"name": "search_course_content", "description": "Search"}]

        # Act
        generator.generate_response(
            query="Question", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert - Second call SHOULD include tools (we're within max rounds)
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs
        assert second_call_kwargs["tools"] == tools


class TestMultiRoundToolCalling:
    """Tests for multi-round sequential tool calling."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_sequential_tool_calls(self, mock_anthropic):
        """Verify Claude can make two sequential tool calls across separate rounds."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Round 1: Claude calls first tool
        round1_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="get_course_outline",
                    tool_input={"course_name": "MCP"},
                    tool_id="tool_1",
                )
            ],
            stop_reason="tool_use",
        )

        # Round 2: Claude calls second tool after seeing first results
        round2_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "lesson 4 topic"},
                    tool_id="tool_2",
                )
            ],
            stop_reason="tool_use",
        )

        # Final: Claude provides answer
        final_response = MockResponse(
            content=[
                MockContentBlock(
                    "text", text="Based on both searches, the answer is..."
                )
            ],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline here",
            "Lesson content here",
        ]

        tools = [
            {"name": "get_course_outline", "description": "Get outline"},
            {"name": "search_course_content", "description": "Search"},
        ]

        # Act
        result = generator.generate_response(
            query="What does lesson 4 cover and find related content",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert result == "Based on both searches, the answer is..."
        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_single_tool_then_text_response(self, mock_anthropic):
        """Verify loop exits when Claude provides text after first tool."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "python"},
                    tool_id="tool_1",
                )
            ],
            stop_reason="tool_use",
        )

        final_response = MockResponse(
            content=[
                MockContentBlock("text", text="Python is a programming language.")
            ],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python course content"

        tools = [{"name": "search_course_content", "description": "Search"}]

        # Act
        result = generator.generate_response(
            query="What is Python?", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert result == "Python is a programming language."
        assert mock_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_forces_completion(self, mock_anthropic):
        """Verify Claude is forced to answer after MAX_TOOL_ROUNDS."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Claude keeps asking for tools
        tool_use_response_1 = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "topic1"},
                    tool_id="tool_1",
                )
            ],
            stop_reason="tool_use",
        )

        tool_use_response_2 = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "topic2"},
                    tool_id="tool_2",
                )
            ],
            stop_reason="tool_use",
        )

        # Would be a 3rd tool use, but max rounds reached so this is the forced final
        forced_final_response = MockResponse(
            content=[MockContentBlock("text", text="Forced answer after max rounds")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [
            tool_use_response_1,
            tool_use_response_2,
            forced_final_response,
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        tools = [{"name": "search_course_content", "description": "Search"}]

        # Act
        result = generator.generate_response(
            query="Complex multi-part question",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert result == "Forced answer after max rounds"
        # 2 tool rounds + 1 forced final = 3 API calls
        assert mock_client.messages.create.call_count == 3
        # 2 tools executed (rounds 1 and 2)
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_final_call_excludes_tools_after_max_rounds(self, mock_anthropic):
        """Verify the final forced call does NOT include tools."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        tool_use_response_1 = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "topic1"},
                    tool_id="tool_1",
                )
            ],
            stop_reason="tool_use",
        )

        tool_use_response_2 = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "topic2"},
                    tool_id="tool_2",
                )
            ],
            stop_reason="tool_use",
        )

        forced_final_response = MockResponse(
            content=[MockContentBlock("text", text="Final answer")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [
            tool_use_response_1,
            tool_use_response_2,
            forced_final_response,
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        tools = [{"name": "search_course_content", "description": "Search"}]

        # Act
        generator.generate_response(
            query="Question", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert - Third call (forced final) should NOT include tools
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs


class TestMessageAccumulation:
    """Tests for message context preservation across tool rounds."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_messages_accumulate_across_rounds(self, mock_anthropic):
        """Verify message list grows correctly across multiple tool rounds."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        round1_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="get_course_outline",
                    tool_input={"course_name": "MCP"},
                    tool_id="tool_1",
                )
            ],
            stop_reason="tool_use",
        )

        round2_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "lesson content"},
                    tool_id="tool_2",
                )
            ],
            stop_reason="tool_use",
        )

        final_response = MockResponse(
            content=[MockContentBlock("text", text="Final answer")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Outline result", "Search result"]

        tools = [
            {"name": "get_course_outline", "description": "Get outline"},
            {"name": "search_course_content", "description": "Search"},
        ]

        # Act
        generator.generate_response(
            query="Original question", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert - Check message accumulation
        # First call: 1 message (user)
        first_call_messages = mock_client.messages.create.call_args_list[0][1][
            "messages"
        ]
        assert len(first_call_messages) == 1

        # Second call: 3 messages (user, assistant tool_use, user tool_result)
        second_call_messages = mock_client.messages.create.call_args_list[1][1][
            "messages"
        ]
        assert len(second_call_messages) == 3

        # Third call: 5 messages (user, asst, result, asst, result)
        third_call_messages = mock_client.messages.create.call_args_list[2][1][
            "messages"
        ]
        assert len(third_call_messages) == 5

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_use_ids_match_in_results(self, mock_anthropic):
        """Verify tool_use_id in results matches the tool_use block."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="unique_tool_id_123",
                )
            ],
            stop_reason="tool_use",
        )

        final_response = MockResponse(
            content=[MockContentBlock("text", text="Answer")], stop_reason="end_turn"
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        tools = [{"name": "search_course_content", "description": "Search"}]

        # Act
        generator.generate_response(
            query="Question", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert - Check tool_use_id matches
        second_call_messages = mock_client.messages.create.call_args_list[1][1][
            "messages"
        ]
        tool_result_message = second_call_messages[2]
        assert tool_result_message["content"][0]["tool_use_id"] == "unique_tool_id_123"


class TestToolLoopErrorHandling:
    """Tests for error handling in the tool loop."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_exception_captured_in_result(self, mock_anthropic):
        """Verify tool exceptions are captured and passed to Claude."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="tool_1",
                )
            ],
            stop_reason="tool_use",
        )

        final_response = MockResponse(
            content=[MockContentBlock("text", text="Sorry, the search failed.")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception(
            "Database connection failed"
        )

        tools = [{"name": "search_course_content", "description": "Search"}]

        # Act
        result = generator.generate_response(
            query="Question", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert - Should still get a response (Claude handles the error)
        assert result == "Sorry, the search failed."

        # Verify error was passed in tool result
        second_call_messages = mock_client.messages.create.call_args_list[1][1][
            "messages"
        ]
        tool_result_content = second_call_messages[2]["content"][0]["content"]
        assert "Tool execution error" in tool_result_content
        assert "Database connection failed" in tool_result_content

    @patch("ai_generator.anthropic.Anthropic")
    def test_continues_after_first_tool_error(self, mock_anthropic):
        """Verify loop continues even if first tool fails, allowing second tool."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # First round: tool fails
        round1_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="get_course_outline",
                    tool_input={"course_name": "NonExistent"},
                    tool_id="tool_1",
                )
            ],
            stop_reason="tool_use",
        )

        # Second round: Claude tries different tool
        round2_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "alternative search"},
                    tool_id="tool_2",
                )
            ],
            stop_reason="tool_use",
        )

        final_response = MockResponse(
            content=[MockContentBlock("text", text="Found alternative info")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_tool_manager = Mock()
        # First tool fails, second succeeds
        mock_tool_manager.execute_tool.side_effect = [
            Exception("Course not found"),
            "Alternative content here",
        ]

        tools = [
            {"name": "get_course_outline", "description": "Get outline"},
            {"name": "search_course_content", "description": "Search"},
        ]

        # Act
        result = generator.generate_response(
            query="Question", tools=tools, tool_manager=mock_tool_manager
        )

        # Assert - Should complete with both tool calls made
        assert result == "Found alternative info"
        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_empty_response_content_handled(self, mock_anthropic):
        """Verify empty text response returns empty string."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Response with no text block
        empty_response = MockResponse(content=[], stop_reason="end_turn")

        mock_client.messages.create.return_value = empty_response

        generator = AIGenerator(api_key="test-key", model="claude-test")

        # Act
        result = generator.generate_response(query="Question")

        # Assert
        assert result == ""
