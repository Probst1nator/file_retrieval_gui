# File Retrieval GUI - Project Index

## 🎯 Project Overview

A sophisticated file management and code retrieval system that combines a modern Tkinter GUI with intelligent file discovery, WebSocket integration, and comprehensive file processing capabilities. The application serves as an intelligent file copier and development assistant, enabling users to efficiently select, organize, and process files within development projects.

## 🏗️ Core Architecture

### **Main Application Layer**
- `main.py` - Application entry point with dual CLI/GUI modes, argument parsing, and getuserfiles tool integration
- `gui.py` - **Primary GUI Application (1,132 lines)**: Comprehensive Tkinter interface with dark theme, WebSocket server, file tree management, drag-and-drop reordering, and multi-selection support
- `smart_paster.py` - **Intelligent File Processing Engine**: Clipboard content parsing, intelligent file discovery, and automated file application with regex-based extraction patterns

## 🏗️ Core Infrastructure

### **Core Components**
- `core/globals.py` - Global state management and configuration storage
- `core/ai_strengths.py` - AI capability definitions and system strengths

## 🛠️ Utility Components

## 🔧 Shared Utilities & Services

### **Core Utilities** (`shared/`)
- `path_resolver.py` - **Path Resolution**: Cross-platform path handling and resolution utilities
- `dia_helper.py` - **Dialog Helpers**: UI dialog and interaction utilities
- `shared_utils_audio.py` - **Audio Processing**: Audio file handling and manipulation utilities

### **Specialized Services** (`shared/utils/`)
- **Search Integration** (`search/`):
  - `brave_search.py` - **Brave Search API**: Web search capabilities for enhanced information retrieval
- **Media Processing** (`youtube/`):
  - `youtube_utils.py` - **YouTube Integration**: Video processing and content extraction utilities

## 🔌 Integration & Tools

### **External Tools** (`tools/`)
- `getuserfiles.py` - **WebSocket File Provider Tool**: Dynamic WebSocket discovery system with multi-tier connection strategy (environment variables → user config → GUI config → port scanning), enabling external tools to retrieve current file selections


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
- **Smart File Discovery**: Intelligent path extraction from text content using regex patterns

### **Configuration Management**
- **Preset System**: Save/load file selection configurations with exclusion patterns
- **Project-Specific Settings**: Per-directory configuration persistence
- **Advanced Exclusions**: Regex-based file filtering with simple and advanced modes