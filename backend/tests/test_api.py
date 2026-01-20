"""Tests for FastAPI endpoints.

This module tests the API endpoints (/api/query, /api/courses, /api/session).
The endpoints are defined inline to avoid import issues with static file mounting
in the main app.py.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Pydantic models (mirroring app.py)
class QueryRequest(BaseModel):
    """Request model for course queries."""
    query: str
    session_id: Optional[str] = None


class Source(BaseModel):
    """Model for a source citation."""
    title: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries."""
    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics."""
    total_courses: int
    course_titles: List[str]


def create_test_app(mock_rag_system):
    """Create a test FastAPI app with mocked RAG system."""
    app = FastAPI(title="Test Course Materials RAG System")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources."""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics."""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        """Clear a conversation session."""
        mock_rag_system.session_manager.clear_session(session_id)
        return {"status": "ok", "session_id": session_id}

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint."""

    def test_query_returns_response_with_sources(self, mock_rag_system):
        """Verify query endpoint returns answer and sources."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?", "session_id": "test-session"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session"
        mock_rag_system.query.assert_called_once_with(
            "What is machine learning?",
            "test-session"
        )

    def test_query_creates_session_when_not_provided(self, mock_rag_system):
        """Verify new session is created when session_id is not provided."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_handles_empty_query(self, mock_rag_system):
        """Verify behavior with empty query string."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "", "session_id": "test-session"}
        )

        # Empty string is valid input - RAG system handles it
        assert response.status_code == 200

    def test_query_returns_500_on_rag_error(self, mock_rag_system):
        """Verify 500 error when RAG system fails."""
        mock_rag_system.query.side_effect = Exception("Database connection failed")
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "Test query", "session_id": "test-session"}
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_query_invalid_json_returns_422(self, mock_rag_system):
        """Verify 422 error for invalid request body."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"invalid_field": "value"}
        )

        assert response.status_code == 422

    def test_query_missing_query_field_returns_422(self, mock_rag_system):
        """Verify 422 error when required 'query' field is missing."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"session_id": "test-session"}
        )

        assert response.status_code == 422

    def test_query_response_sources_format(self, mock_rag_system):
        """Verify sources in response have correct structure."""
        mock_rag_system.query.return_value = (
            "Answer text",
            [
                {"title": "Course A - Lesson 1", "url": "https://example.com/a1"},
                {"title": "Course B - Lesson 2", "url": None}
            ]
        )
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "Test", "session_id": "test-session"}
        )

        assert response.status_code == 200
        sources = response.json()["sources"]
        assert len(sources) == 2
        assert sources[0]["title"] == "Course A - Lesson 1"
        assert sources[0]["url"] == "https://example.com/a1"
        assert sources[1]["url"] is None


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint."""

    def test_courses_returns_statistics(self, mock_rag_system):
        """Verify courses endpoint returns course count and titles."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2
        assert "Python Course" in data["course_titles"]
        assert "ML Course" in data["course_titles"]

    def test_courses_empty_catalog(self, mock_rag_system):
        """Verify response when no courses are loaded."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_returns_500_on_error(self, mock_rag_system):
        """Verify 500 error when analytics fails."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Storage error")
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Storage error" in response.json()["detail"]


class TestSessionEndpoint:
    """Tests for DELETE /api/session/{session_id} endpoint."""

    def test_clear_session_returns_ok(self, mock_rag_system):
        """Verify session clearing returns success."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.delete("/api/session/test-session-456")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["session_id"] == "test-session-456"
        mock_rag_system.session_manager.clear_session.assert_called_once_with(
            "test-session-456"
        )

    def test_clear_nonexistent_session(self, mock_rag_system):
        """Verify clearing nonexistent session still returns ok."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.delete("/api/session/nonexistent-session")

        # Session manager doesn't raise on nonexistent session
        assert response.status_code == 200


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_check_returns_healthy(self, mock_rag_system):
        """Verify health check returns healthy status."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestRequestValidation:
    """Tests for request validation and error handling."""

    def test_query_with_extra_fields_ignored(self, mock_rag_system):
        """Verify extra fields in request are ignored (Pydantic behavior)."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={
                "query": "Test query",
                "session_id": "test-session",
                "extra_field": "should be ignored"
            }
        )

        assert response.status_code == 200

    def test_query_with_null_session_id(self, mock_rag_system):
        """Verify null session_id triggers session creation."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "Test query", "session_id": None}
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == "test-session-123"

    def test_query_content_type_required(self, mock_rag_system):
        """Verify request requires JSON content type."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post(
            "/api/query",
            data="query=test",  # Form data instead of JSON
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        assert response.status_code == 422


class TestConcurrentRequests:
    """Tests for handling concurrent requests."""

    def test_multiple_queries_different_sessions(self, mock_rag_system):
        """Verify multiple queries with different sessions work correctly."""
        mock_rag_system.query.side_effect = [
            ("Answer 1", [{"title": "Source 1", "url": None}]),
            ("Answer 2", [{"title": "Source 2", "url": None}])
        ]
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response1 = client.post(
            "/api/query",
            json={"query": "Query 1", "session_id": "session-1"}
        )
        response2 = client.post(
            "/api/query",
            json={"query": "Query 2", "session_id": "session-2"}
        )

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["answer"] == "Answer 1"
        assert response2.json()["answer"] == "Answer 2"
        assert mock_rag_system.query.call_count == 2
