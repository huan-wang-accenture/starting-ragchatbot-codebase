"""Shared pytest fixtures for RAG system tests."""
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "/tmp/test_chroma"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-test"
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store with common search behavior."""
    store = Mock()
    store.search.return_value = SearchResults(
        documents=["Test content about machine learning."],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1]
    )
    store.get_lesson_link.return_value = "https://example.com/lesson1"
    store.get_course_metadata.return_value = {
        "title": "Test Course",
        "course_link": "https://example.com/course",
        "lesson_count": 2,
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Introduction"},
            {"lesson_number": 2, "lesson_title": "Basics"}
        ]
    }
    return store


@pytest.fixture
def empty_search_results():
    """Create empty search results."""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def sample_search_results():
    """Create sample search results with multiple documents."""
    return SearchResults(
        documents=[
            "Machine learning is a subset of artificial intelligence.",
            "Python is commonly used for machine learning.",
            "Neural networks are a type of ML model."
        ],
        metadata=[
            {"course_title": "ML Fundamentals", "lesson_number": 1},
            {"course_title": "ML Fundamentals", "lesson_number": 2},
            {"course_title": "Deep Learning", "lesson_number": 1}
        ],
        distances=[0.1, 0.2, 0.3]
    )


@pytest.fixture
def sample_course():
    """Create a sample course with lessons."""
    return Course(
        title="Test Course",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Introduction",
                lesson_link="https://example.com/lesson1"
            ),
            Lesson(
                lesson_number=2,
                title="Getting Started",
                lesson_link="https://example.com/lesson2"
            )
        ]
    )


@pytest.fixture
def sample_chunks():
    """Create sample course chunks for testing."""
    return [
        CourseChunk(
            content="This is the first chunk about Python basics.",
            course_title="Python Course",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="This chunk covers Python variables and types.",
            course_title="Python Course",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="This chunk is about Python functions.",
            course_title="Python Course",
            lesson_number=2,
            chunk_index=0
        )
    ]


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = Mock()
    return client


@pytest.fixture
def mock_ai_generator(mock_anthropic_client):
    """Create a mock AI generator."""
    generator = Mock()
    generator.generate_response.return_value = "This is a test response."
    return generator


@pytest.fixture
def mock_rag_system(mock_vector_store, mock_ai_generator):
    """Create a mock RAG system."""
    rag = Mock()
    rag.query.return_value = (
        "Test response about the topic.",
        [{"title": "Test Course - Lesson 1", "url": "https://example.com/lesson1"}]
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Course", "ML Course"]
    }
    rag.session_manager = Mock()
    rag.session_manager.create_session.return_value = "test-session-123"
    rag.session_manager.clear_session.return_value = None
    return rag


class MockContentBlock:
    """Mock for Anthropic content blocks."""
    def __init__(self, block_type, text=None, tool_name=None, tool_input=None, tool_id=None):
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


@pytest.fixture
def mock_text_response():
    """Create a mock text response from Claude."""
    return MockResponse(
        content=[MockContentBlock("text", text="This is a direct answer.")],
        stop_reason="end_turn"
    )


@pytest.fixture
def mock_tool_use_response():
    """Create a mock tool use response from Claude."""
    return MockResponse(
        content=[
            MockContentBlock(
                "tool_use",
                tool_name="search_course_content",
                tool_input={"query": "machine learning"},
                tool_id="tool_123"
            )
        ],
        stop_reason="tool_use"
    )
