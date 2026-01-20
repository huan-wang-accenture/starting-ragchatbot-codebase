"""Tests for RAGSystem end-to-end query handling."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from config import Config


class TestMaxResultsConfiguration:
    """Critical tests for MAX_RESULTS configuration issue."""

    def test_config_max_results_not_zero(self):
        """
        CRITICAL TEST: Verify MAX_RESULTS is not zero.

        This is the root cause of 'query failed' errors.
        MAX_RESULTS=0 causes ChromaDB to return no results.
        """
        # Act
        from config import config

        # Assert - MAX_RESULTS should be > 0 for search to work
        assert config.MAX_RESULTS > 0, (
            f"MAX_RESULTS is {config.MAX_RESULTS}, but must be > 0 "
            "for searches to return results. This is likely the cause "
            "of 'query failed' errors."
        )

    def test_vector_store_uses_max_results(self):
        """Verify VectorStore respects max_results parameter."""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create VectorStore with max_results=0 (the bug)
            store_with_zero = VectorStore(
                chroma_path=temp_dir,
                embedding_model="all-MiniLM-L6-v2",
                max_results=0
            )

            # Act - Search with default limit (uses max_results)
            result = store_with_zero.search(query="test query")

            # Assert - With max_results=0, should get empty results
            assert result.is_empty(), (
                "Expected empty results with max_results=0, "
                "but got results. VectorStore may not be using max_results correctly."
            )

    def test_vector_store_returns_results_when_configured_properly(self):
        """Verify VectorStore returns results with proper configuration."""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create VectorStore with proper max_results
            store = VectorStore(
                chroma_path=temp_dir,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5
            )

            # Add test content
            from models import CourseChunk
            test_chunks = [
                CourseChunk(
                    content="Machine learning is a subset of artificial intelligence.",
                    course_title="ML Course",
                    lesson_number=1,
                    chunk_index=0
                ),
                CourseChunk(
                    content="Python is a popular programming language for ML.",
                    course_title="ML Course",
                    lesson_number=2,
                    chunk_index=1
                )
            ]
            store.add_course_content(test_chunks)

            # Act
            result = store.search(query="machine learning")

            # Assert - Should return results
            assert not result.is_empty(), (
                "Expected results with max_results=5 and content in store, "
                "but got empty results."
            )
            assert len(result.documents) > 0


class TestVectorStoreSearch:
    """Tests for VectorStore search functionality."""

    def test_search_returns_search_results_object(self):
        """Verify search returns SearchResults dataclass."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(
                chroma_path=temp_dir,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5
            )

            result = store.search(query="test")

            assert isinstance(result, SearchResults)
            assert hasattr(result, 'documents')
            assert hasattr(result, 'metadata')
            assert hasattr(result, 'distances')
            assert hasattr(result, 'error')

    def test_search_with_course_filter(self):
        """Verify course name filtering works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(
                chroma_path=temp_dir,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5
            )

            # Add content from multiple courses
            from models import Course, Lesson, CourseChunk

            # Add course metadata first (provide non-None values to avoid ChromaDB issues)
            course1 = Course(
                title="Python Course",
                course_link="https://example.com/python",
                instructor="John Doe",
                lessons=[Lesson(lesson_number=1, title="Intro", lesson_link="https://example.com/python/1")]
            )
            course2 = Course(
                title="Java Course",
                course_link="https://example.com/java",
                instructor="Jane Doe",
                lessons=[Lesson(lesson_number=1, title="Intro", lesson_link="https://example.com/java/1")]
            )
            store.add_course_metadata(course1)
            store.add_course_metadata(course2)

            # Add course content
            chunks = [
                CourseChunk(
                    content="Python variables and data types",
                    course_title="Python Course",
                    lesson_number=1,
                    chunk_index=0
                ),
                CourseChunk(
                    content="Java variables and data types",
                    course_title="Java Course",
                    lesson_number=1,
                    chunk_index=1
                )
            ]
            store.add_course_content(chunks)

            # Act - Search with Python filter
            result = store.search(query="variables", course_name="Python")

            # Assert - Should only return Python course content
            assert not result.is_empty()
            for meta in result.metadata:
                assert meta["course_title"] == "Python Course"

    def test_search_nonexistent_course(self):
        """Verify behavior when filtering by course name with no courses in store."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(
                chroma_path=temp_dir,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5
            )

            # Don't add any courses - search for nonexistent course in empty catalog
            # This ensures _resolve_course_name returns None

            # Act - Search with filter for nonexistent course
            result = store.search(query="test", course_name="Nonexistent Course")

            # Assert - With no courses in catalog, should get error
            assert result.error is not None
            assert "No course found" in result.error


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method."""

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_query_returns_response_and_sources(self, mock_vector_store_class, mock_ai_gen_class):
        """Verify query returns both response and sources."""
        # Arrange
        mock_store = Mock()
        mock_vector_store_class.return_value = mock_store

        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "This is the answer."
        mock_ai_gen_class.return_value = mock_ai_gen

        # Create a mock config
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "/tmp/test_chroma"
        mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-test"
        mock_config.MAX_HISTORY = 2

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        # Setup tool manager to return sources
        rag.search_tool.last_sources = [
            {"title": "Course - Lesson 1", "url": "https://example.com"}
        ]

        # Act
        response, sources = rag.query("What is Python?")

        # Assert
        assert response == "This is the answer."
        assert len(sources) == 1
        assert sources[0]["title"] == "Course - Lesson 1"

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_query_passes_tools_to_ai_generator(self, mock_vector_store_class, mock_ai_gen_class):
        """Verify query passes tool definitions to AI generator."""
        # Arrange
        mock_store = Mock()
        mock_vector_store_class.return_value = mock_store

        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Response"
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "/tmp/test_chroma"
        mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-test"
        mock_config.MAX_HISTORY = 2

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        # Act
        rag.query("Question?")

        # Assert - Check tools were passed
        call_kwargs = mock_ai_gen.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) == 2  # search and outline tools

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_query_resets_sources_after_retrieval(self, mock_vector_store_class, mock_ai_gen_class):
        """Verify sources are reset after each query."""
        # Arrange
        mock_store = Mock()
        mock_vector_store_class.return_value = mock_store

        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Response"
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_config = Mock()
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "/tmp/test_chroma"
        mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-test"
        mock_config.MAX_HISTORY = 2

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        # Setup sources
        rag.search_tool.last_sources = [{"title": "Test", "url": None}]

        # Act
        rag.query("Question?")

        # Assert - Sources should be reset
        assert rag.search_tool.last_sources == []


class TestSearchResultsDataclass:
    """Tests for SearchResults dataclass."""

    def test_from_chroma_creates_results(self):
        """Verify from_chroma factory method works."""
        chroma_results = {
            'documents': [["Doc 1", "Doc 2"]],
            'metadatas': [[{"course": "A"}, {"course": "B"}]],
            'distances': [[0.1, 0.2]]
        }

        result = SearchResults.from_chroma(chroma_results)

        assert result.documents == ["Doc 1", "Doc 2"]
        assert len(result.metadata) == 2
        assert result.distances == [0.1, 0.2]
        assert result.error is None

    def test_empty_creates_error_results(self):
        """Verify empty factory method creates error results."""
        result = SearchResults.empty("Test error message")

        assert result.documents == []
        assert result.metadata == []
        assert result.distances == []
        assert result.error == "Test error message"

    def test_is_empty_returns_true_for_no_documents(self):
        """Verify is_empty works correctly."""
        empty_result = SearchResults(documents=[], metadata=[], distances=[])
        non_empty_result = SearchResults(
            documents=["content"],
            metadata=[{}],
            distances=[0.1]
        )

        assert empty_result.is_empty() is True
        assert non_empty_result.is_empty() is False
