# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) chatbot that enables users to query course materials and receive context-aware responses powered by Anthropic's Claude AI. Built with Python FastAPI backend and vanilla JavaScript frontend.

## Commands

**Important: Always use `uv` for running Python files and dependency management. Do not use `pip` or `python` directly.**

```bash
# Run any Python file
uv run python <file.py>
```

```bash
# Install dependencies
uv sync

# Run the application
./run.sh

# Manual backend startup
cd backend && uv run uvicorn app:app --reload --port 8000

# Access points after running:
# - Web UI: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

## Architecture

### Backend (`backend/`)

The backend follows a modular component architecture:

- **`app.py`** - FastAPI application with API endpoints (`/api/query`, `/api/courses`)
- **`rag_system.py`** - Main orchestrator that coordinates all components
- **`document_processor.py`** - Parses course documents and chunks text (800 chars, 100 overlap)
- **`vector_store.py`** - ChromaDB integration with SentenceTransformer embeddings (all-MiniLM-L6-v2)
- **`ai_generator.py`** - Claude API integration with tool calling support
- **`search_tools.py`** - Tool definitions for Claude's `search_course_content` tool
- **`session_manager.py`** - In-memory conversation history management
- **`config.py`** - Configuration constants (model, chunk sizes, paths)
- **`models.py`** - Pydantic models (Course, Lesson, CourseChunk)

### Frontend (`frontend/`)

Vanilla JavaScript chat interface that communicates with the backend API.

### Query Flow

1. Frontend sends `{ query, session_id }` to `/api/query`
2. `RAGSystem.query()` wraps the query and calls `AIGenerator.generate_response()`
3. Claude API (1st call) receives the query with `tool_choice="auto"` and decides whether to search
4. If Claude returns `stop_reason: "tool_use"`, `AIGenerator._handle_tool_execution()` executes the tool
5. `CourseSearchTool.execute()` calls `VectorStore.search()` which queries ChromaDB
6. Claude API (2nd call) receives search results and synthesizes the final answer
7. Sources are extracted via `ToolManager.get_last_sources()` and returned to frontend

### Document Indexing (Startup)

1. `app.py` startup event loads documents from `docs/` folder
2. `DocumentProcessor.process_course_document()` extracts metadata and chunks content
3. `VectorStore.add_course_metadata()` stores course info in `course_catalog` collection
4. `VectorStore.add_course_content()` stores chunks in `course_content` collection

### Document Format (`docs/`)

Course text files follow a specific structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [url]
[content...]

Lesson 1: Getting Started
[content...]
```

The processor uses regex to detect `Lesson N:` markers and splits content accordingly.

## Environment Setup

Requires `.env` file with:
```
ANTHROPIC_API_KEY=your_key_here
```

Copy from `.env.example` as a starting point.

## Key Configuration (`backend/config.py`)

- Claude model: `claude-sonnet-4-20250514`
- Embedding model: `all-MiniLM-L6-v2`
- Chunk size: 800 characters with 100 overlap
- Max search results: 5
- Max conversation history: 2 messages

## ChromaDB Collections

- **`course_catalog`**: Course metadata (title, instructor, links) - used for fuzzy course name resolution
- **`course_content`**: Document chunks with `course_title`, `lesson_number`, `chunk_index` metadata
