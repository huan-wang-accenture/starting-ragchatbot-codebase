import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of tool calling rounds per query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **search_course_content**: Use for questions about specific course content or detailed educational materials
- **get_course_outline**: Use for questions about course structure, lesson lists, or what topics a course covers
  - Always include: course title, course link, and the complete lesson list with lesson numbers and titles

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Up to two tool calls per query** - Use sequential calls when needed (e.g., get outline first, then search specific content)
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Course outline/structure questions**: Use get_course_outline tool
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
  - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Supports up to MAX_TOOL_ROUNDS sequential tool calls, allowing Claude
        to reason about results from one tool before making additional calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        # Use tool loop if tools are provided
        if tools:
            return self._run_tool_loop(messages, system_content, tools, tool_manager)

        # No tools - single API call
        response = self._make_api_call(messages, system_content)
        return self._extract_text_response(response)

    def _extract_text_response(self, response) -> str:
        """
        Extract text content from Claude API response.

        Args:
            response: The Anthropic API response object

        Returns:
            Text content string, or empty string if no text block found
        """
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        system_content: str,
        tools: Optional[List] = None,
        include_tools: bool = True,
    ):
        """
        Make a single API call to Claude.

        Args:
            messages: Message list for the API call
            system_content: System prompt content
            tools: Available tools (optional)
            include_tools: Whether to include tools in the request

        Returns:
            Raw Anthropic API response
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }
        if tools and include_tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        return self.client.messages.create(**api_params)

    def _execute_tools(self, response, tool_manager) -> List[Dict[str, Any]]:
        """
        Execute all tool calls from a response.

        Args:
            response: The Anthropic API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            List of tool_result dicts for message building
        """
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                except Exception as e:
                    result = f"Tool execution error: {str(e)}"
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
        return tool_results

    def _build_messages_with_tool_results(
        self,
        messages: List[Dict[str, Any]],
        assistant_response,
        tool_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Append assistant response and tool results to message list.

        Args:
            messages: Current message list
            assistant_response: The API response containing tool_use blocks
            tool_results: List of tool_result dicts

        Returns:
            New message list with appended content (does not mutate original)
        """
        new_messages = messages.copy()
        new_messages.append(
            {"role": "assistant", "content": assistant_response.content}
        )
        new_messages.append({"role": "user", "content": tool_results})
        return new_messages

    def _run_tool_loop(
        self,
        messages: List[Dict[str, Any]],
        system_content: str,
        tools: List,
        tool_manager,
    ) -> str:
        """
        Iterative tool execution loop supporting multiple rounds.

        Args:
            messages: Initial message list
            system_content: System prompt content
            tools: Available tools
            tool_manager: Manager to execute tools

        Returns:
            Final text response from Claude
        """
        current_round = 0
        current_messages = messages.copy()

        while current_round < self.MAX_TOOL_ROUNDS:
            response = self._make_api_call(
                current_messages, system_content, tools, include_tools=True
            )

            # Check if Claude wants to use tools
            if response.stop_reason != "tool_use":
                return self._extract_text_response(response)

            # No tool_manager means we can't execute tools
            if not tool_manager:
                return self._extract_text_response(response)

            # Execute tools and build updated messages
            tool_results = self._execute_tools(response, tool_manager)
            current_messages = self._build_messages_with_tool_results(
                current_messages, response, tool_results
            )
            current_round += 1

        # Max rounds reached - force completion without tools
        final_response = self._make_api_call(
            current_messages, system_content, tools=None, include_tools=False
        )
        return self._extract_text_response(final_response)
