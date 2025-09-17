from typing import Optional
from flask import Flask, render_template_string, jsonify, request
import threading
import webbrowser
import queue
import requests
import time
from core.chat import Chat, Role
import subprocess
import os
from multiprocessing import Process

class WebServer:
    def __init__(self, port: int = 5000):
        self.port = port
        self.app = Flask(__name__)
        self.chat: Optional[Chat] = None
        self.server_process: Optional[Process] = None
        self.notification_thread: Optional[threading.Thread] = None
        self.notification_queue: queue.Queue = queue.Queue()
        self.server_ready = threading.Event()
        # Create a queue for print statements
        self.print_queue: queue.Queue = queue.Queue()
        self.setup_routes()  # Call setup_routes in initialization
        self._start_notification_thread()
        self._start_print_thread()
        
    def _start_notification_thread(self):
        """Start a background thread to handle notifications"""
        def notification_worker():
            while True:
                try:
                    role, content = self.notification_queue.get()
                    if role is None:  # Sentinel value to stop the thread
                        break
                    try:
                        # Wait for server to be ready before sending notifications
                        if self.server_ready.wait(timeout=5):  # Wait up to 5 seconds
                            # Make the request in a non-blocking way
                            requests.post(
                                f'http://localhost:{self.port}/add_message',
                                json={'role': role.value, 'content': content},
                                timeout=1
                            )
                    except (requests.exceptions.RequestException, ConnectionError):
                        # Ignore connection errors - they're expected when the server isn't running
                        pass
                except Exception:
                    # Ignore any other errors to keep the thread running
                    pass
                
        self.notification_thread = threading.Thread(target=notification_worker, daemon=True)
        self.notification_thread.start()
        
    def _start_print_thread(self):
        """Start a background thread to handle print messages"""
        def print_worker():
            while True:
                try:
                    content = self.print_queue.get()
                    if content is None:  # Sentinel value to stop the thread
                        break
                    try:
                        # Add as system message with a log indicator
                        self.add_message_to_chat(Role.SYSTEM, f"📃 Log: {content}")
                    except Exception:
                        # Ignore any errors to keep the thread running
                        pass
                except Exception:
                    # Ignore any other errors to keep the thread running
                    pass
                
        self.print_thread = threading.Thread(target=print_worker, daemon=True)
        self.print_thread.start()
        
    def log_print(self, *args, **kwargs):
        """Log a print statement to the web interface"""
        # Reconstruct the print arguments into a string, similar to how print works
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        content = sep.join(str(arg) for arg in args) + end
        
        # Add to queue for processing
        self.print_queue.put(content.strip())
        
    def setup_routes(self):
        @self.app.route('/')
        def home():
            return render_template_string(self.get_html_template())
        
        @self.app.route('/add_message', methods=['POST'])
        def add_message():
            """Endpoint to add a new message"""
            data = request.get_json()
            if not data or 'role' not in data or 'content' not in data:
                return jsonify({'error': 'Invalid message format'}), 400
            
            # Add message to chat if it doesn't exist already
            if self.chat:
                role = Role[data['role'].upper()] if isinstance(data['role'], str) else data['role']
                self.chat.add_message(role, data['content'])
            return jsonify({'status': 'ok'})
        
        @self.app.route('/messages')
        def get_messages():
            """Get all messages in the chat"""
            if not self.chat:
                print("Debug: Chat instance is None")
                return jsonify([])
            
            if not self.chat.messages:
                print("Debug: No messages in chat")
                return jsonify([])
            
            messages = []
            for role, content in self.chat.messages:
                # Convert role to string value if it's an enum
                role_str = role.value if isinstance(role, Role) else str(role)
                messages.append({
                    'role': role_str,
                    'content': content
                })
            
            print(f"Debug: Returning {len(messages)} messages")
            return jsonify(messages)
        
        @self.app.route('/health')
        def health_check():
            """Endpoint to verify server is running"""
            return jsonify({'status': 'ok'})
    
    def add_message_to_chat(self, role: Role, content: str):
        """Add a message to the chat and notify the web interface"""
        if self.chat:
            # Ensure the role is properly converted to Role enum if it's a string
            if isinstance(role, str):
                role = Role[role.upper()]
            
            # Add message to chat
            self.chat.add_message(role, content)
            
            # Queue the notification to be sent asynchronously
            self.notification_queue.put((role, content))
            
            # Save chat to JSON after each message
            self.chat.save_to_json()
            
    def get_html_template(self) -> str:
        return '''
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .message { margin: 10px; padding: 10px; border-radius: 5px; white-space: pre-wrap; }
        .user { background: #e3f2fd; margin-left: 20%; }
        .assistant { background: #f5f5f5; margin-right: 20%; }
        .system { 
            background: #f0f8ff; 
            border-left: 4px solid #4682b4; 
            font-family: monospace; 
            font-size: 0.9em;
            margin-right: 20%;
            opacity: 0.85;
        }
        .tool-call { 
            background: #fff3e0; 
            margin-right: 20%;
            border-left: 4px solid #ff9800;
        }
        .tool-header {
            font-weight: bold;
            color: #e65100;
            margin-bottom: 8px;
        }
        .tool-content {
            margin-left: 24px;
        }
        pre {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        code { font-family: monospace; }
        #toggle-logs {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 8px 15px;
            background-color: #4682b4;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .log-hidden .system {
            display: none;
        }
    </style>
</head>
<body>
    <div id="chat-container"></div>
    <button id="toggle-logs">Toggle Logs</button>
    <script>
        let showLogs = true;
        
        document.getElementById('toggle-logs').addEventListener('click', function() {
            showLogs = !showLogs;
            document.body.classList.toggle('log-hidden', !showLogs);
        });
        
        function formatContent(content) {
            // Handle code blocks
            if (content.includes('```')) {
                var parts = content.split('```');
                var result = parts[0];
                for (var i = 1; i < parts.length; i++) {
                    if (i % 2 === 1) {
                        // This is a code block
                        var codeContent = parts[i];
                        var language = '';
                        
                        // Check if there's a language specified
                        if (codeContent.indexOf('\\n') > 0) {
                            language = codeContent.substring(0, codeContent.indexOf('\\n')).trim();
                            if (language) {
                                codeContent = codeContent.substring(codeContent.indexOf('\\n')+1);
                            }
                        }
                        
                        result += '<pre><code>' + codeContent + '</code></pre>';
                    } else {
                        result += parts[i];
                    }
                }
                return result;
            }
            return content;
        }

        function addMessage(role, content) {
            var div = document.createElement('div');
            div.className = 'message ' + role;
            
            // Format tool calls specially
            if (content.includes('🛠️ Using tool:')) {
                div.className += ' tool-call';
                var lines = content.split('\\n');
                var toolHeader = lines[0];  // First line with the tool icon
                var toolContent = lines.slice(1).join('\\n');  // Rest of the content
                
                div.innerHTML = '<div class="tool-header">' + toolHeader + '</div>' +
                               '<div class="tool-content">' + formatContent(toolContent) + '</div>';
            } else {
                div.innerHTML = formatContent(content);
            }
            
            document.getElementById('chat-container').appendChild(div);
            
            // Scroll to bottom
            window.scrollTo(0, document.body.scrollHeight);
        }

        function loadMessages() {
            fetch('/messages')
                .then(function(r) { return r.json(); })
                .then(function(messages) {
                    var container = document.getElementById('chat-container');
                    container.innerHTML = '';
                    messages.forEach(function(m) {
                        addMessage(m.role, m.content);
                    });
                });
        }

        loadMessages();
        setInterval(loadMessages, 1000);
    </script>
</body>
</html>
'''
    
    def start(self, chat: Optional[Chat] = None) -> None:
        """Start the web server"""
        self.chat = chat if chat else Chat()
        
        # Save the app to a temporary file that gunicorn can import
        with open('app.py', 'w') as f:
            f.write('''
from shared.utils.web.web_server import WebServer
from core.chat import Chat
from core.globals import g
import builtins

# Create and configure the server
server = WebServer()
server.chat = Chat.load_from_json()  # Load the chat from the saved JSON file
app = server.app

# Store the web server in globals for print redirection
g.web_server = server

# Patch the built-in print function to redirect to the web UI
def custom_print(*args, **kwargs):
    # Call the original print function
    g.original_print(*args, **kwargs)
    
    # Also send to the web interface
    try:
        server.log_print(*args, **kwargs)
    except Exception:
        pass
        
# Replace the built-in print function
builtins.print = custom_print

# Print a test message to confirm the web UI is receiving logs
print("Web server initialized successfully - all print() outputs will now appear in the web UI")
''')
        
        # Save current chat state so gunicorn can load it
        if self.chat:
            self.chat.save_to_json()
        
        # Start gunicorn in a separate process
        self.server_process = subprocess.Popen([
            'gunicorn',
            '--bind', f'127.0.0.1:{self.port}',
            '--workers', '1',
            '--threads', '4',
            '--timeout', '120',
            'app:app'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to be ready before opening browser
        if self.wait_for_server(timeout=5):
            self.server_ready.set()
            print(f"Web UI is ready at http://localhost:{self.port} - all print() outputs will also appear in the web UI")
            # Open browser in a non-blocking way
            threading.Thread(
                target=lambda: webbrowser.open(f'http://localhost:{self.port}'),
                daemon=True
            ).start()
        else:
            print(f"Warning: Web server failed to start on port {self.port}")
            # Print server output for debugging
            if self.server_process and self.server_process.stderr:
                print("Server error output:")
                print(self.server_process.stderr.read().decode())
    
    def stop(self):
        """Stop the web server and notification thread"""
        if self.notification_thread:
            self.notification_queue.put((None, None))  # Send sentinel value to stop the thread
            self.notification_thread.join(timeout=1)
            self.notification_thread = None
        if hasattr(self, 'print_thread'):
            self.print_queue.put(None)  # Send sentinel value to stop the thread
            self.print_thread.join(timeout=1)
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
        try:
            os.remove('app.py')
        except:
            pass
    
    def wait_for_server(self, timeout: int = 10) -> bool:
        """Wait for server to be ready, returns True if server is ready, False if timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f'http://localhost:{self.port}/health', timeout=1)
                if response.status_code == 200:
                    return True
            except (requests.exceptions.RequestException, ConnectionError):
                time.sleep(0.1)
        return False 

    def _run_server(self):
        """Run the Flask server"""
        self.app.run(host='127.0.0.1', port=self.port, debug=False, use_reloader=False) 