# File Retrieval GUI - Project Index

## 🎯 Project Overview

A sophisticated file management and code retrieval system that combines a modern Tkinter GUI with AI-powered file discovery, WebSocket integration, and comprehensive LLM provider support. The application serves as an intelligent file copier and code assistant, enabling users to efficiently select, organize, and process files within development projects.

## 🏗️ Core Architecture

### **Main Application Layer**
- `main.py` - Application entry point with dual CLI/GUI modes, argument parsing, and getuserfiles tool integration
- `gui.py` - **Primary GUI Application (1,132 lines)**: Comprehensive Tkinter interface with dark theme, WebSocket server, file tree management, drag-and-drop reordering, and multi-selection support
- `smart_paster.py` - **Intelligent File Processing Engine**: Clipboard content parsing, AI-powered file discovery, and automated file application with regex-based extraction patterns

### **AI-Powered Components**
- `ai_path_finder.py` - **AI File Discovery System**: LLM-powered code block location finder with multi-strategy approach (simple direct prompts and structured JSON responses)

## 🧠 LLM Infrastructure (Core Module)

### **Router & Management**
- `core/llm_router.py` - **Singleton LLM Router**: Multi-provider request routing, load balancing, token counting, streaming support, and provider fallback mechanisms
- `core/chat.py` - **Conversation Management**: Message handling with role-based typing, debug window support, and terminal color formatting
- `core/llm.py` - LLM configuration and model management utilities
- `core/globals.py` - Global state management and configuration storage
- `core/ai_strengths.py` - AI capability definitions and provider-specific strengths

### **Multi-Provider Support** (`core/providers/`)
- `cls_anthropic_interface.py` - **Anthropic Claude Integration**: Full API support with streaming, thinking budget, and temperature controls
- `cls_openai_interface.py` - **OpenAI GPT Integration**: Complete API implementation with streaming and configuration management
- `cls_google_interface.py` - **Google Gemini Integration**: Google AI platform connectivity with model-specific handling
- `cls_groq_interface.py` - **Groq API Integration**: High-speed inference with timeout and rate limit exception handling
- `cls_nvidia_interface.py` - **NVIDIA API Integration**: Enterprise-grade LLM access through NVIDIA's platform
- `cls_ollama_interface.py` - **Local Ollama Integration**: Self-hosted model support for private deployments
- `cls_human_as_interface.py` - **Human-in-the-Loop Interface**: Manual response injection for testing and fallback scenarios
- `cls_whisper_interface.py` - **Audio Transcription**: Speech-to-text capabilities using OpenAI Whisper

## 🛠️ Infrastructure Components

### **Rate Limiting System** (`infrastructure/rate_limiting/`)
- `cls_rate_limit_tracker.py` - **Core Rate Limit Management**: Singleton pattern tracker for model cooldowns and quota management
- `cls_enhanced_rate_limit_tracker.py` - **Advanced Rate Limiting**: Enhanced tracking with detailed metrics and recovery strategies
- `cls_rate_limit_config.py` - **Configuration Management**: Rate limit settings and provider-specific configurations
- `cls_rate_limit_parsers.py` - **Response Parsing**: API response analysis for rate limit detection and parameter extraction

## 🔧 Shared Utilities & Services

### **Core Utilities** (`shared/`)
- `common_utils.py` - **Multi-Purpose Utilities**: Audio processing, PDF handling, SQLite operations, speech recognition, and file operations
- `path_resolver.py` - **Path Resolution**: Cross-platform path handling and resolution utilities
- `cmd_execution.py` - **Command Execution**: Safe subprocess management and command running utilities
- `dia_helper.py` - **Dialog Helpers**: UI dialog and interaction utilities
- `utils_audio.py` - **Audio Processing**: Audio file handling and manipulation utilities

### **Specialized Services** (`shared/utils/`)
- **Web Services** (`web/`):
  - `web_server.py` - **Flask Web Server**: Multi-threaded web interface with notification system and process management
- **Search Integration** (`search/`):
  - `brave_search.py` - **Brave Search API**: Web search capabilities for enhanced information retrieval
- **Content Processing** (`rag/`):
  - `rag_utils.py` - **RAG (Retrieval-Augmented Generation)**: Document processing and knowledge base integration
- **Media Processing** (`youtube/`):
  - `youtube_utils.py` - **YouTube Integration**: Video processing and content extraction utilities
- **Development Tools** (`python/`):
  - `python_utils.py` - **Python Development**: Code analysis, execution, and development utilities
- **Audio Processing** (`audio/`):
  - `audio_utils.py` - **Advanced Audio**: Enhanced audio processing and manipulation capabilities

## 🔌 Integration & Tools

### **External Tools** (`tools/`)
- `getuserfiles.py` - **WebSocket File Provider Tool**: Dynamic WebSocket discovery system with multi-tier connection strategy (environment variables → user config → GUI config → port scanning), enabling external tools to retrieve current file selections

### **Interface Abstractions** (`py_classes/`)
- `unified_interfaces.py` - **Provider Interface Definitions**: Abstract base classes for AI providers, audio interfaces, and service abstractions with comprehensive logging configuration

## 🧪 Testing & Development

### **Test Suite**
- `test_*.py` (8 files) - **Comprehensive Testing**: WebSocket integration tests, file provider negotiation tests, handler fix validation, and API connectivity tests
- `demo_file_provider_negotiation.py` - **Integration Demos**: Live demonstration of WebSocket communication and file provider capabilities

## 🎨 GUI Features & Capabilities

### **Advanced File Management**
- **Multi-Selection Support**: Ctrl/Shift key combinations for bulk file selection and deletion
- **Drag & Drop Reordering**: Visual file organization with intuitive reordering
- **Smart Filtering**: Real-time file filtering with regex support and directory exclusions
- **Project Tree Navigation**: Expandable/collapsible directory structure with lazy loading

### **WebSocket Integration**
- **Real-time Communication**: Built-in WebSocket server for external tool integration
- **Client Management**: Connection tracking, status monitoring, and broadcast capabilities
- **Dynamic Port Discovery**: Automatic port selection with fallback mechanisms

### **Content Processing**
- **Preview System**: Real-time content preview with character/line counting
- **Clipboard Integration**: Direct copy-to-clipboard functionality with size optimization
- **Apply Changes Feature**: Automated file modification from structured text input (supports markdown code blocks)

### **Configuration Management**
- **Preset System**: Save/load file selection configurations with exclusion patterns
- **Project-Specific Settings**: Per-directory configuration persistence
- **Advanced Exclusions**: Regex-based file filtering with simple and advanced modes