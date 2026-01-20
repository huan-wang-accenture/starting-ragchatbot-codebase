"""Tests for CourseSearchTool and related search functionality."""

import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Tests for CourseSearchTool.execute() method."""

    def test_execute_returns_results_when_content_exists(self):
        """Verify search returns formatted results when content is found."""
        # Arrange
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=["This is content about machine learning basics."],
            metadata=[{"course_title": "ML Course", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = CourseSearchTool(mock_store)

        # Act
        result = tool.execute(query="machine learning")

        # Assert
        assert "ML Course" in result
        assert "Lesson 1" in result
        assert "machine learning basics" in result
        mock_store.search.assert_called_once_with(
            query="machine learning", course_name=None, lesson_number=None
        )

    def test_execute_with_course_filter(self):
        """Test filtering by course name passes filter to vector store."""
        # Arrange
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=["Python content"],
            metadata=[{"course_title": "Python Course", "lesson_number": 2}],
            distances=[0.2],
        )
        mock_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_store)

        # Act
        result = tool.execute(query="variables", course_name="Python")

        # Assert
        mock_store.search.assert_called_once_with(
            query="variables", course_name="Python", lesson_number=None
        )
        assert "Python Course" in result

    def test_execute_with_lesson_filter(self):
        """Test filtering by lesson number passes filter to vector store."""
        # Arrange
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.15],
        )
        mock_store.get_lesson_link.return_value = "https://example.com/lesson3"

        tool = CourseSearchTool(mock_store)

        # Act
        result = tool.execute(query="testing", lesson_number=3)

        # Assert
        mock_store.search.assert_called_once_with(
            query="testing", course_name=None, lesson_number=3
        )
        assert "Lesson 3" in result

    def test_execute_empty_results(self):
        """Verify proper 'no content found' message when results are empty."""
        # Arrange
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool = CourseSearchTool(mock_store)

        # Act
        result = tool.execute(query="nonexistent topic")

        # Assert
        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(self):
        """Verify filter info is included in 'no content found' message."""
        # Arrange
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool = CourseSearchTool(mock_store)

        # Act
        result = tool.execute(
            query="topic", course_name="Specific Course", lesson_number=5
        )

        # Assert
        assert "No relevant content found" in result
        assert "Specific Course" in result
        assert "lesson 5" in result

    def test_execute_handles_vector_store_errors(self):
        """Test error handling when vector store returns an error."""
        # Arrange
        mock_store = Mock()
        mock_store.search.return_value = SearchResults.empty(
            "Search error: Database connection failed"
        )

        tool = CourseSearchTool(mock_store)

        # Act
        result = tool.execute(query="test query")

        # Assert
        assert "Search error" in result
        assert "Database connection failed" in result

    def test_format_results_includes_sources(self):
        """Verify last_sources is populated after search."""
        # Arrange
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )
        mock_store.get_lesson_link.side_effect = [
            "https://example.com/a1",
            "https://example.com/a2",
        ]

        tool = CourseSearchTool(mock_store)

        # Act
        tool.execute(query="test")

        # Assert
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["title"] == "Course A - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/a1"
        assert tool.last_sources[1]["title"] == "Course A - Lesson 2"

    def test_format_results_deduplicates_sources(self):
        """Verify duplicate sources are not included multiple times."""
        # Arrange
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=["Content 1", "Content 2", "Content 3"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 1},  # Duplicate
                {"course_title": "Course A", "lesson_number": 2},
            ],
            distances=[0.1, 0.15, 0.2],
        )
        mock_store.get_lesson_link.return_value = "https://example.com"

        tool = CourseSearchTool(mock_store)

        # Act
        tool.execute(query="test")

        # Assert - Should only have 2 unique sources, not 3
        assert len(tool.last_sources) == 2

    def test_tool_definition_format(self):
        """Verify get_tool_definition returns correct Anthropic format."""
        # Arrange
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)

        # Act
        definition = tool.get_tool_definition()

        # Assert
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool.execute() method."""

    def test_execute_returns_outline(self):
        """Verify outline is properly formatted."""
        # Arrange
        mock_store = Mock()
        mock_store.get_course_metadata.return_value = {
            "title": "Test Course",
            "course_link": "https://example.com/course",
            "lesson_count": 2,
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Introduction"},
                {"lesson_number": 2, "lesson_title": "Basics"},
            ],
        }

        tool = CourseOutlineTool(mock_store)

        # Act
        result = tool.execute(course_name="Test")

        # Assert
        assert "Test Course" in result
        assert "https://example.com/course" in result
        assert "Lesson 1: Introduction" in result
        assert "Lesson 2: Basics" in result

    def test_execute_course_not_found(self):
        """Verify error message when course not found."""
        # Arrange
        mock_store = Mock()
        mock_store.get_course_metadata.return_value = None

        tool = CourseOutlineTool(mock_store)

        # Act
        result = tool.execute(course_name="Nonexistent")

        # Assert
        assert "No course found" in result
        assert "Nonexistent" in result


class TestToolManager:
    """Tests for ToolManager class."""

    def test_register_and_execute_tool(self):
        """Verify tools can be registered and executed."""
        # Arrange
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = None

        manager = ToolManager()
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        # Act
        result = manager.execute_tool("search_course_content", query="test")

        # Assert
        assert "Test" in result

    def test_execute_unknown_tool(self):
        """Verify error message for unknown tool."""
        # Arrange
        manager = ToolManager()

        # Act
        result = manager.execute_tool("unknown_tool")

        # Assert
        assert "not found" in result

    def test_get_tool_definitions(self):
        """Verify all registered tool definitions are returned."""
        # Arrange
        mock_store = Mock()
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        manager.register_tool(CourseOutlineTool(mock_store))

        # Act
        definitions = manager.get_tool_definitions()

        # Assert
        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_get_last_sources(self):
        """Verify sources can be retrieved from tools."""
        # Arrange
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = "https://example.com"

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        manager.register_tool(search_tool)

        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test")

        # Act
        sources = manager.get_last_sources()

        # Assert
        assert len(sources) == 1
        assert sources[0]["title"] == "Course - Lesson 1"

    def test_reset_sources(self):
        """Verify sources can be reset."""
        # Arrange
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = None

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        manager.register_tool(search_tool)

        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test")

        # Act
        manager.reset_sources()

        # Assert
        assert manager.get_last_sources() == []
