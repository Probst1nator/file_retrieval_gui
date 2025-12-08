# tools/file_copier/gui.py
import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
from typing import Dict, List, Optional
import threading
import asyncio
import fnmatch
import signal
import re
from datetime import datetime

# Third-party libraries
from dotenv import load_dotenv
import websockets
try:
    import pyperclip
except ImportError:
    pyperclip = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

try:
    from ctypes import windll  # type: ignore
except (ImportError, AttributeError):
    windll = None # Define as None on non-Windows systems

# Local application imports
# Must modify path before importing from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from smart_paster import apply_changes_to_files, preview_changes_to_files, apply_selected_changes, IGNORE_DIRS, build_clipboard_content, process_smart_request, FileChange, ChangeType

# Load environment variables after imports
load_dotenv()

# High DPI scaling for Windows
if sys.platform == "win32" and windll:
    try:
        windll.shcore.SetProcessDpiAwareness(1)
    except AttributeError:
        # Might be an older version of Windows
        pass


# --- NEW: WebSocket Server Configuration ---
# Support environment variables with fallback to defaults
WEBSOCKET_HOST = os.getenv("WEBSOCKET_HOST", "localhost")
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", "8765"))
WEBSOCKET_PORT_RANGE_START = 8765
WEBSOCKET_PORT_RANGE_END = 8769
# --- END NEW ---

DEFAULT_PRESET_NAME = "default"
CONFIG_FILENAME = ".file_copier_config.json"

DARK_BG, DARK_FG, DARK_SELECT_BG = "#2b2b2b", "#ffffff", "#404040"
DARK_ENTRY_BG, DARK_BUTTON_BG, DARK_TREE_BG = "#3c3c3c", "#404040", "#2b2b2b"
DARK_ERROR_FG = "#F44336" # Red for errors/unsupported files

def is_text_file(filepath: str) -> bool:
    try:
        with open(filepath, 'rb') as f:
            return b'\0' not in f.read(1024)
    except (IOError, PermissionError):
        return False

def is_includable_file(filepath: str) -> bool:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.pdf', '.odt'}:
        return True
    return is_text_file(filepath)

def get_script_directory() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

def format_filesize(size_bytes: int) -> str:
    """Format file size adaptively (KB/MB/GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def format_timestamp(timestamp: float) -> str:
    """Format timestamp as absolute datetime with second precision."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

class FileCopierApp:
    def __init__(self, root: tk.Tk, directory: str):
        self.root = root
        self.directory = os.path.abspath(directory)
        self.config_file_path = os.path.join(get_script_directory(), CONFIG_FILENAME)
        self._initialize_state()
        self._initialize_websocket_state()
        self._setup_styles()
        self._create_widgets()
        self._bind_events()
        self._setup_interrupt_handler()
        self._log_message("Initializing...")
        self.load_project_config()
        self.root.after(100, self.start_async_project_load)
        self.root.after(500, self._delayed_websocket_start)

    def _initialize_state(self):
        self.selected_files_map: Dict[str, bool] = {}
        self.file_metadata: Dict[str, Dict] = {}  # Store raw metadata for sorting
        self.preview_visible = False
        self.all_text_files: List[str] = []
        self._search_job: Optional[str] = None
        self._auto_save_job: Optional[str] = None
        self.full_config: Dict[str, Dict] = {}
        self.presets: Dict[str, Dict] = {}
        self.drag_start_item: Optional[str] = None
        self.supported_binary_ext = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.pdf', '.odt'}
        self.sort_column: Optional[str] = None
        self.sort_reverse: bool = False
        self.drag_enabled: bool = True

    def _initialize_websocket_state(self):
        self.websocket_server = None  # type: ignore
        self.websocket_loop: Optional[asyncio.AbstractEventLoop] = None
        self.connected_clients = set()
        # FIXED: Corrected type hint for websockets
        self.client_info = {}  # type: ignore
        self.current_shared_string = "File Retrieval GUI is running, but no files have been selected yet."
        self.websocket_enabled = True
        self.websocket_start_time: Optional[datetime] = None
        self.connections_refresh_job: Optional[str] = None
        self.websocket_host = WEBSOCKET_HOST
        self.websocket_port = WEBSOCKET_PORT
        self.actual_websocket_port: Optional[int] = None

    def _start_websocket_server(self):
        if not self.websocket_enabled:
            self._log_message("WebSocket server is disabled.", "info")
            return
        
        self._cleanup_websocket_state()
            
        try:
            asyncio.run(self._async_websocket_server_main())
        except OSError as e:
            self._log_message(f"WebSocket Error: Could not start server. Port {WEBSOCKET_PORT} may be in use. {e}", "error")
        except Exception as e:
            self._log_message(f"An unexpected error occurred in the WebSocket server thread: {e}", "error")
        finally:
            self._cleanup_websocket_state()

    async def _async_websocket_server_main(self):
        self.websocket_loop = asyncio.get_running_loop()
        server_started = False
        last_error = None
        
        ports_to_try = list(dict.fromkeys([self.websocket_port] + list(range(WEBSOCKET_PORT_RANGE_START, WEBSOCKET_PORT_RANGE_END + 1))))
        
        for port in ports_to_try:
            try:
                self._log_message(f"Attempting to start WebSocket server on {self.websocket_host}:{port}")
                self.websocket_server = await websockets.serve(self._websocket_handler, self.websocket_host, port)
                self.actual_websocket_port = port
                self.websocket_start_time = datetime.now()
                self._log_message(f"WebSocket server started successfully on ws://{self.websocket_host}:{port}", "success")
                if port != self.websocket_port:
                    self._log_message(f"Note: Using port {port} instead of configured port {self.websocket_port}", "info")
                self._save_websocket_config()
                server_started = True
                break
            except OSError as e:
                last_error = e
                self._log_message(f"Port {port} unavailable, trying next port...", "warning")
        
        if not server_started:
            error_msg = f"Could not start WebSocket server on any port in range {WEBSOCKET_PORT_RANGE_START}-{WEBSOCKET_PORT_RANGE_END}"
            if last_error:
                error_msg += f". Last error: {last_error}"
            raise OSError(error_msg)
        
        assert self.websocket_server is not None
        await self.websocket_server.wait_closed()

    # FIXED: Corrected type hint for websockets
    async def _websocket_handler(self, websocket):  # type: ignore
        client_address = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}" if websocket.remote_address else "unknown"
        connect_time = datetime.now()
        
        self.connected_clients.add(websocket)
        self.client_info[websocket] = {
            'address': client_address,
            'connect_time': connect_time,
            'last_activity': connect_time
        }
        
        self._log_message(f"New client connected from {client_address} ({len(self.connected_clients)} total).", "info")
        try:
            await websocket.send(self.current_shared_string)
            async for _ in websocket:
                if websocket in self.client_info:
                    self.client_info[websocket]['last_activity'] = datetime.now()
        except websockets.exceptions.ConnectionClosed:
            self._log_message(f"Client {client_address} connection closed.", "info")
        except Exception as e:
            self._log_message(f"Error with client {client_address}: {e}", "error")
        finally:
            self.connected_clients.discard(websocket)
            if websocket in self.client_info:
                del self.client_info[websocket]
            self._log_message(f"Client {client_address} removed ({len(self.connected_clients)} total).", "info")

    async def _broadcast_update(self):
        if not self.connected_clients:
            return
        
        clients_to_send = list(self.connected_clients)
        message = self.current_shared_string
        
        tasks = [client.send(message) for client in clients_to_send]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        failed_clients = [clients_to_send[i] for i, res in enumerate(results) if isinstance(res, Exception)]
        for client in failed_clients:
            self.connected_clients.discard(client)
        
        if len(clients_to_send) > len(failed_clients):
             self._log_message(f"Broadcasted update to {len(clients_to_send) - len(failed_clients)} client(s).")

    def _get_selected_files_ordered(self) -> List[str]:
        return [self.selected_files_tree.item(item, "values")[0] for item in self.selected_files_tree.get_children("")]

    def _update_and_broadcast_string(self):
        selected_files = self._get_selected_files_ordered()
        if not selected_files:
            self.current_shared_string = "No files selected."
        else:
            self.current_shared_string = build_clipboard_content(
                [os.path.join(self.directory, f) for f in selected_files],
                self.directory
            )
        self._safe_broadcast_update()

    def _update_ui_state(self, auto_save: bool = True):
        self.update_selected_count()
        self.update_preview()
        if auto_save:
            self._debounce_auto_save()
        self._update_and_broadcast_string()

    def on_closing(self):
        self.auto_save_current_preset()
        if self.connections_refresh_job:
            self.root.after_cancel(self.connections_refresh_job)
        if self.websocket_server:
            self._log_message("Stopping WebSocket server...")
            self._stop_websocket_server()
        self.root.destroy()
        
    def update_preview(self):
        if not self.preview_visible:
            return
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        out = self.current_shared_string
        self.preview_text.insert(1.0, out)
        
        if out == "No files selected.":
            self.preview_stats_var.set("L: 0 | C: 0")
        else:
            self.preview_stats_var.set(f"L: {len(out.splitlines()):,} | C: {len(out):,}")
        self.preview_text.see(1.0)
        self.preview_text.config(state=tk.DISABLED)

    def _setup_styles(self):
        self.root.title(f"File Retrieval GUI - {os.path.basename(self.directory)}")

        # Set window class for proper taskbar identification
        try:
            self.root.tk.call('wm', 'class', self.root._w, 'FileRetrievalGUI')
        except Exception:
            pass

        # Set window icon - load multiple resolutions for crisp display
        icon_png_path = os.path.join(get_script_directory(), "assets", "Icon.png")

        if os.path.exists(icon_png_path) and Image and ImageTk:
            try:
                # Load original PNG
                original = Image.open(icon_png_path)

                # Crop transparent padding to make icon larger in taskbar
                if original.mode == 'RGBA':
                    # Get bounding box of non-transparent content
                    bbox = original.getbbox()
                    if bbox:
                        original = original.crop(bbox)

                # Provide full range of icon sizes - taskbar picks what it needs
                # Most taskbars use 32-64px, but we provide more for flexibility
                icon_sizes = [256, 128, 96, 64, 48, 32, 24, 16]
                icon_images = []

                for size in icon_sizes:
                    # Maintain aspect ratio by fitting into square canvas
                    img_copy = original.copy()
                    img_copy.thumbnail((size, size), Image.Resampling.LANCZOS)

                    # Create square transparent canvas
                    square_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))

                    # Center the icon on the canvas
                    offset = ((size - img_copy.width) // 2, (size - img_copy.height) // 2)
                    square_img.paste(img_copy, offset, img_copy if img_copy.mode == 'RGBA' else None)

                    photo = ImageTk.PhotoImage(square_img)
                    icon_images.append(photo)

                # Pass all sizes to iconphoto (first arg True = apply to future windows too)
                self.root.iconphoto(True, *icon_images)

                # Keep references to prevent garbage collection
                self.icon_images = icon_images
            except Exception as e:
                print(f"Warning: Could not load icon: {e}")

        self.root.geometry("1400x900")
        self.root.configure(bg=DARK_BG)
        style = ttk.Style()
        base_font = ("Segoe UI", 10) if sys.platform == "win32" else ("Helvetica", 11)

        style.theme_use('clam')

        self.root.option_add('*TCombobox*Listbox.background', DARK_ENTRY_BG)
        self.root.option_add('*TCombobox*Listbox.foreground', DARK_FG)
        self.root.option_add('*TCombobox*Listbox.selectBackground', DARK_SELECT_BG)
        self.root.option_add('*TCombobox*Listbox.selectForeground', DARK_FG)
        self.root.option_add('*TCombobox*Listbox.font', base_font)
        self.root.option_add('*TCombobox*Listbox.selectBorderWidth', '0')

        style.configure('.', font=base_font, background=DARK_BG, foreground=DARK_FG)
        style.configure("TFrame", background=DARK_BG)
        style.configure("TLabel", background=DARK_BG, foreground=DARK_FG)

        style.configure("TCombobox", fieldbackground=DARK_ENTRY_BG, background=DARK_ENTRY_BG, foreground=DARK_FG, bordercolor=DARK_SELECT_BG, insertcolor=DARK_FG, arrowcolor=DARK_FG)
        style.map("TCombobox", foreground=[('readonly', DARK_FG)], fieldbackground=[('readonly', DARK_ENTRY_BG)], background=[('readonly', DARK_ENTRY_BG)])
        
        style.configure("TEntry", fieldbackground=DARK_ENTRY_BG, background=DARK_ENTRY_BG, foreground=DARK_FG, bordercolor=DARK_SELECT_BG, insertcolor=DARK_FG)
        style.configure("TButton", background=DARK_BUTTON_BG, foreground=DARK_FG, bordercolor=DARK_SELECT_BG, padding=5)
        style.configure("Treeview", background=DARK_TREE_BG, foreground=DARK_FG, fieldbackground=DARK_TREE_BG, rowheight=25)
        style.map("Treeview", background=[('selected', DARK_SELECT_BG)])
        style.configure("TCheckbutton", background=DARK_BG, foreground=DARK_FG)
        style.configure('Accent.TButton', font=(base_font[0], base_font[1], "bold"), background="#0078d4", foreground=DARK_FG)
        style.map('Accent.TButton', background=[('active', '#106ebe')])
        style.configure("TNotebook", background=DARK_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=DARK_BUTTON_BG, foreground=DARK_FG, padding=[8, 4])
        style.map("TNotebook.Tab", background=[("selected", DARK_SELECT_BG)], expand=[("selected", [1, 1, 1, 0])])

    def _create_widgets(self):
        self.main_container = ttk.Frame(self.root, padding=10)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        vertical_pane = ttk.PanedWindow(self.main_container, orient=tk.VERTICAL)
        vertical_pane.pack(fill=tk.BOTH, expand=True)
        self.top_pane = ttk.PanedWindow(vertical_pane, orient=tk.HORIZONTAL)
        vertical_pane.add(self.top_pane, weight=4)
        bottom_pane_container = ttk.Frame(vertical_pane)
        vertical_pane.add(bottom_pane_container, weight=2)
        
        self._create_tree_pane()
        self._create_selection_pane()
        self._create_bottom_notebook(bottom_pane_container)

        bottom_controls_frame = ttk.Frame(self.main_container)
        bottom_controls_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        self.btn_toggle_preview = ttk.Button(bottom_controls_frame, text="Show Preview", command=self.toggle_preview)
        self.btn_toggle_preview.pack(side=tk.LEFT)
        self.btn_copy = ttk.Button(bottom_controls_frame, text="Copy to Clipboard", command=self.copy_to_clipboard, style='Accent.TButton')
        self.btn_copy.pack(side=tk.RIGHT)

        self.preview_frame = ttk.Frame(self.main_container)
        self._create_preview_widgets()

    def _create_tree_pane(self, *args):
        tree_frame = ttk.Frame(self.top_pane, padding=(0, 0, 5, 0))
        search_frame = ttk.Frame(tree_frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(search_frame, text="Filter:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        exclusion_main_frame = ttk.Frame(tree_frame)
        exclusion_main_frame.pack(fill=tk.X, pady=(0, 10))
        self.advanced_exclude_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(exclusion_main_frame, text="Advanced Exclusions (Regex)", variable=self.advanced_exclude_var, command=self._toggle_exclude_mode).pack(anchor='w')
        self.simple_exclude_frame = ttk.Frame(exclusion_main_frame)
        ttk.Label(self.simple_exclude_frame, text="Exclude Dirs:").grid(row=0, column=0, sticky='w', pady=(5, 0))
        self.exclude_dirs_var = tk.StringVar(value=" ".join(sorted(list(IGNORE_DIRS))))
        self.exclude_dirs_entry = ttk.Entry(self.simple_exclude_frame, textvariable=self.exclude_dirs_var)
        self.exclude_dirs_entry.grid(row=0, column=1, sticky='ew', pady=(5, 0), padx=(5, 0))
        ttk.Label(self.simple_exclude_frame, text="Exclude Files:").grid(row=1, column=0, sticky='w')
        self.exclude_patterns_var = tk.StringVar(value="*.log *.json *.csv *.env .DS_Store .gitignore")
        self.exclude_patterns_entry = ttk.Entry(self.simple_exclude_frame, textvariable=self.exclude_patterns_var)
        self.exclude_patterns_entry.grid(row=1, column=1, sticky='ew', padx=(5, 0))
        self.simple_exclude_frame.grid_columnconfigure(1, weight=1)
        self.advanced_exclude_frame = ttk.Frame(exclusion_main_frame)
        ttk.Label(self.advanced_exclude_frame, text="Exclude (regex):").pack(side=tk.LEFT)
        self.exclusion_var = tk.StringVar(value=r"venv/|\.git/|\.idea/|\.vscode/|__pycache__|/node_modules/|/build/|/dist/|.*\.log$")
        self.exclusion_entry = ttk.Entry(self.advanced_exclude_frame, textvariable=self.exclusion_var)
        self.exclusion_entry.pack(fill=tk.X, expand=True)
        tree_controls = ttk.Frame(tree_frame)
        tree_controls.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(tree_controls, text="Add Folder", command=self.add_selected_folder).pack(side=tk.LEFT)
        ttk.Button(tree_controls, text="Add All Visible", command=self.add_all_visible_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(tree_controls, text="Expand All", command=self.expand_all_tree_items).pack(side=tk.LEFT)
        ttk.Button(tree_controls, text="Collapse All", command=self.collapse_all_tree_items).pack(side=tk.LEFT, padx=5)
        self.tree = ttk.Treeview(tree_frame, show="tree headings")
        self.tree.heading("#0", text="Project Structure", anchor='w')
        ysb = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=ysb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.top_pane.add(tree_frame, weight=2)
        self.tree.insert("", "end", text="Scanning project...", tags=('info',))
        self.tree.tag_configure('info', foreground='#888888')

    def _create_selection_pane(self, *args):
        selection_frame = ttk.Frame(self.top_pane, padding=(5, 0, 0, 0))
        preset_frame = ttk.Frame(selection_frame)
        preset_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar()
        self.preset_combobox = ttk.Combobox(preset_frame, textvariable=self.preset_var, state="readonly")
        self.preset_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(preset_frame, text="Save As...", command=self.save_current_as_preset, width=10).pack(side=tk.LEFT)
        ttk.Button(preset_frame, text="Remove", command=self.remove_selected_preset, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(selection_frame, text="Selected Files (Drag to Reorder)", font=("Segoe UI", 10, "bold")).pack(pady=(0, 5), anchor='w')
        
        selection_tree_frame = ttk.Frame(selection_frame)
        selection_tree_frame.pack(fill=tk.BOTH, expand=True)
        selection_tree_frame.grid_rowconfigure(0, weight=1)
        selection_tree_frame.grid_columnconfigure(0, weight=1)
        
        self.selected_files_tree = ttk.Treeview(selection_tree_frame, columns=("filepath", "filetype", "filesize", "char_count", "changed"), show="headings", selectmode="extended")
        self.selected_files_tree.heading("filepath", text="File Path")
        self.selected_files_tree.heading("filetype", text="Type", anchor='center')
        self.selected_files_tree.heading("filesize", text="Size", anchor='e')
        self.selected_files_tree.heading("char_count", text="Characters", anchor='e')
        self.selected_files_tree.heading("changed", text="Modified", anchor='e')
        self.selected_files_tree.column("filepath", width=250, stretch=tk.YES)
        self.selected_files_tree.column("filetype", width=60, stretch=tk.NO, anchor='center')
        self.selected_files_tree.column("filesize", width=80, stretch=tk.NO, anchor='e')
        self.selected_files_tree.column("char_count", width=100, stretch=tk.NO, anchor='e')
        self.selected_files_tree.column("changed", width=140, stretch=tk.NO, anchor='e')
        
        self.selected_files_tree.tag_configure('unsupported', foreground=DARK_ERROR_FG)

        tree_scrollbar = ttk.Scrollbar(selection_tree_frame, orient='vertical', command=self.selected_files_tree.yview)
        self.selected_files_tree.configure(yscrollcommand=tree_scrollbar.set)

        self.selected_files_tree.grid(row=0, column=0, sticky="nsew")
        tree_scrollbar.grid(row=0, column=1, sticky="ns")

        # Set focus to enable keyboard shortcuts
        self.selected_files_tree.focus_set()

        controls_frame = ttk.Frame(selection_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="Remove Selected", command=self.remove_selected).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        self.selected_count_var = tk.StringVar(value="0 files selected")
        ttk.Label(controls_frame, textvariable=self.selected_count_var).pack(side=tk.RIGHT)
        self.top_pane.add(selection_frame, weight=3)

    def _create_bottom_notebook(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        tools_tab = ttk.Frame(notebook, padding=5)
        notebook.add(tools_tab, text="Tools")
        self._create_tools_pane(tools_tab)
        log_tab = ttk.Frame(notebook, padding=5)
        notebook.add(log_tab, text="Log")
        self._create_log_pane(log_tab)
        connections_tab = ttk.Frame(notebook, padding=5)
        notebook.add(connections_tab, text="Connections")
        self._create_connections_pane(connections_tab)

    def _create_tools_pane(self, parent):
        tools_container = ttk.Frame(parent)
        tools_container.pack(fill=tk.BOTH, expand=True)

        smart_paster_frame = ttk.Frame(tools_container)
        smart_paster_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        ttk.Label(smart_paster_frame, text="Filepath Extractor", font=("Segoe UI", 10, "bold")).pack(anchor='w')
        self.smart_paste_text = scrolledtext.ScrolledText(smart_paster_frame, height=4, wrap=tk.WORD, bg=DARK_ENTRY_BG, fg=DARK_FG, insertbackground=DARK_FG, font=("Segoe UI", 10), borderwidth=0, highlightthickness=1)
        self.smart_paste_text.pack(fill=tk.BOTH, expand=True, pady=5)
        smart_paster_controls = ttk.Frame(smart_paster_frame)
        smart_paster_controls.pack(fill=tk.X)
        ttk.Button(smart_paster_controls, text="Find & Select Files", command=self._initiate_smart_paste, style='Accent.TButton').pack(side=tk.LEFT)

        apply_changes_frame = ttk.Frame(tools_container)
        apply_changes_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(apply_changes_frame, text="Apply Changes to Files", font=("Segoe UI", 10, "bold")).pack(anchor='w')
        self.apply_changes_text = scrolledtext.ScrolledText(apply_changes_frame, height=4, wrap=tk.WORD, bg=DARK_ENTRY_BG, fg=DARK_FG, insertbackground=DARK_FG, font=("Segoe UI", 10), borderwidth=0, highlightthickness=1)
        self.apply_changes_text.pack(fill=tk.BOTH, expand=True, pady=5)
        apply_changes_controls = ttk.Frame(apply_changes_frame)
        apply_changes_controls.pack(fill=tk.X)
        ttk.Button(apply_changes_controls, text="Preview Changes", command=self._initiate_preview_changes).pack(side=tk.LEFT)
        ttk.Button(apply_changes_controls, text="Apply to Files", command=self._initiate_apply_changes, style='Accent.TButton').pack(side=tk.LEFT, padx=(5, 0))

    def _create_log_pane(self, parent):
        ttk.Label(parent, text="Global Log", font=("Segoe UI", 10, "bold")).pack(anchor='w', pady=(5, 2))
        self.log_text = scrolledtext.ScrolledText(parent, height=5, wrap=tk.WORD, bg=DARK_ENTRY_BG, fg=DARK_FG, font=("Consolas", 9), borderwidth=0, highlightthickness=1)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.tag_config("success", foreground="#4CAF50")
        self.log_text.tag_config("error", foreground="#F44336")
        self.log_text.tag_config("info", foreground="#FFFFFF")
        self.log_text.tag_config("warning", foreground="#FFC107")
        self.log_text.config(state=tk.DISABLED)
    
    def _create_connections_pane(self, parent):
        connections_container = ttk.Frame(parent)
        connections_container.pack(fill=tk.BOTH, expand=True)
        
        status_frame = ttk.Frame(connections_container)
        status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(status_frame, text="WebSocket Server Status", font=("Segoe UI", 10, "bold")).pack(anchor='w', pady=(0, 5))
        
        status_info_frame = ttk.Frame(status_frame)
        status_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.server_status_var = tk.StringVar(value="Stopped")
        self.server_port_var = tk.StringVar(value=f"Port: {WEBSOCKET_PORT}")
        self.server_uptime_var = tk.StringVar(value="Uptime: --")
        self.client_count_var = tk.StringVar(value="Connected: 0")
        
        ttk.Label(status_info_frame, text="Status:").grid(row=0, column=0, sticky='w')
        self.status_label = ttk.Label(status_info_frame, textvariable=self.server_status_var, foreground="#F44336")
        self.status_label.grid(row=0, column=1, sticky='w', padx=(5, 0))
        
        ttk.Label(status_info_frame, textvariable=self.server_port_var).grid(row=1, column=0, columnspan=2, sticky='w')
        ttk.Label(status_info_frame, textvariable=self.server_uptime_var).grid(row=2, column=0, columnspan=2, sticky='w')
        ttk.Label(status_info_frame, textvariable=self.client_count_var).grid(row=3, column=0, columnspan=2, sticky='w')
        
        controls_frame = ttk.Frame(status_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_toggle_server = ttk.Button(controls_frame, text="Disable Server", command=self.toggle_websocket_server)
        self.btn_toggle_server.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_restart_server = ttk.Button(controls_frame, text="Restart Server", command=self.restart_websocket_server)
        self.btn_restart_server.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_disconnect_all = ttk.Button(controls_frame, text="Disconnect All", command=self.disconnect_all_clients)
        self.btn_disconnect_all.pack(side=tk.LEFT)
        
        clients_frame = ttk.Frame(connections_container)
        clients_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(clients_frame, text="Connected Clients", font=("Segoe UI", 10, "bold")).pack(anchor='w', pady=(0, 5))
        
        clients_list_frame = ttk.Frame(clients_frame)
        clients_list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.clients_listbox = tk.Listbox(clients_list_frame, bg=DARK_ENTRY_BG, fg=DARK_FG, selectbackground=DARK_SELECT_BG, 
                                         font=("Consolas", 9), highlightthickness=0, borderwidth=0)
        clients_scrollbar = ttk.Scrollbar(clients_list_frame, orient='vertical', command=self.clients_listbox.yview)
        self.clients_listbox.configure(yscrollcommand=clients_scrollbar.set, exportselection=False)
        
        self.clients_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        clients_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self._refresh_connections_display()
        
    def _create_preview_widgets(self):
        preview_header_frame = ttk.Frame(self.preview_frame)
        preview_header_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(preview_header_frame, text="Preview", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.preview_stats_var = tk.StringVar()
        ttk.Label(preview_header_frame, textvariable=self.preview_stats_var, foreground="#aaaaaa").pack(side=tk.RIGHT)
        self.preview_text = scrolledtext.ScrolledText(self.preview_frame, height=10, wrap=tk.WORD, bg=DARK_ENTRY_BG, fg=DARK_FG, font=("Consolas", 10), borderwidth=0, highlightthickness=1)
        self.preview_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    def _bind_events(self):
        self.tree.bind("<Double-1>", self.on_tree_double_click)
        self.tree.bind('<<TreeviewOpen>>', self.on_tree_expand)
        self.selected_files_tree.bind("<Double-1>", lambda e: self.remove_selected())
        self.selected_files_tree.bind("<Button-1>", self.on_drag_start)
        self.selected_files_tree.bind("<B1-Motion>", self.on_drag_motion)
        self.selected_files_tree.bind("<<TreeviewSelect>>", lambda e: self.update_preview())
        self.selected_files_tree.bind("<Delete>", lambda e: self.remove_selected())
        self.selected_files_tree.bind("<Key-Delete>", lambda e: self.remove_selected())
        self.selected_files_tree.bind("<Control-a>", self.select_all_files)
        self.selected_files_tree.bind("<Command-a>", self.select_all_files)
        # Bind column header clicks for sorting
        for col in ["filepath", "filetype", "filesize", "char_count", "changed"]:
            self.selected_files_tree.heading(col, command=lambda c=col: self._on_column_click(c))
        self.preset_combobox.bind("<<ComboboxSelected>>", self.on_preset_selected)
        self.search_var.trace_add("write", self._debounce_search)
        self.exclude_dirs_var.trace_add("write", self._debounce_search)
        self.exclude_patterns_var.trace_add("write", self._debounce_search)
        self.exclusion_var.trace_add("write", self._debounce_search)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        for widget in [self.search_entry, self.exclusion_entry, self.exclude_dirs_entry, self.exclude_patterns_entry, self.preview_text, self.apply_changes_text]:
            self._bind_select_all(widget)
            
    def _log_message(self, message: str, level: str = 'info'):
        def update_log():
            if self.root.winfo_exists():
                self.log_text.config(state=tk.NORMAL)
                timestamp = datetime.now().strftime('%H:%M:%S')
                self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", (level,))
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
        if hasattr(self, 'root') and self.root:
            self.root.after(0, update_log)
    
    def _initiate_apply_changes(self):
        content = self.apply_changes_text.get("1.0", tk.END).strip()
        if not content:
            self._log_message("Apply Changes: Textbox is empty.", 'warning')
            return
        self._log_message("Apply Changes: Starting file writing operation...")
        threading.Thread(target=self._apply_changes_worker, args=(content,), daemon=True).start()

    def _apply_changes_worker(self, content: str):
        results = apply_changes_to_files(content, self.directory)
        for file_path in results["success"]:
            self._log_message(f"  ✓ Applied changes to {file_path}", 'success')
        for error_msg in results["errors"]:
            self._log_message(f"  ✗ {error_msg}", 'error')
        summary = f"Apply Changes: Finished. {len(results['success'])} success, {len(results['errors'])} errors."
        self._log_message(summary, 'error' if results['errors'] else 'success')
        if results['success']:
            self.root.after(0, self._perform_filter)

    def _initiate_preview_changes(self):
        """Show the preview changes dialog."""
        content = self.apply_changes_text.get("1.0", tk.END).strip()
        if not content:
            self._log_message("Preview Changes: Textbox is empty.", 'warning')
            return
        
        self._log_message("Preview Changes: Parsing file changes...")
        try:
            changes = preview_changes_to_files(content, self.directory)
            if not changes:
                self._log_message("Preview Changes: No valid file blocks found.", 'warning')
                return
            
            self._log_message(f"Preview Changes: Found {len(changes)} file changes.")
            
            # Show the preview dialog
            dialog = PreviewChangesDialog(self.root, changes, self.directory)
            
            if dialog.result == 'apply_selected':
                # Apply the selected changes
                selected_changes = [c for c in changes if c.selected]
                if selected_changes:
                    self._log_message(f"Applying {len(selected_changes)} selected changes...")
                    threading.Thread(target=self._apply_selected_changes_worker, args=(selected_changes,), daemon=True).start()
                else:
                    self._log_message("No changes selected for application.", 'warning')
            
        except Exception as e:
            self._log_message(f"Preview Changes Error: {e}", 'error')

    def _apply_selected_changes_worker(self, changes: List[FileChange]):
        """Apply the selected changes in a worker thread."""
        results = apply_selected_changes(changes)
        for file_path in results["success"]:
            self._log_message(f"  ✓ Applied changes to {file_path}", 'success')
        for error_msg in results["errors"]:
            self._log_message(f"  ✗ {error_msg}", 'error')
        summary = f"Apply Selected Changes: Finished. {len(results['success'])} success, {len(results['errors'])} errors."
        self._log_message(summary, 'error' if results['errors'] else 'success')
        if results['success']:
            self.root.after(0, self._perform_filter)

    def _initiate_smart_paste(self):
        content = self.smart_paste_text.get("1.0", tk.END).strip()
        if not content:
            self._log_message("Smart Paster: Textbox is empty.", 'warning')
            return
        self._log_message("Smart Paster: Starting file discovery...")
        self.smart_paste_text.config(state=tk.DISABLED)
        threading.Thread(target=self._smart_paste_worker, args=(content,), daemon=True).start()

    def _smart_paste_worker(self, content: str):
        try:
            found_rel_paths = asyncio.run(process_smart_request(content, self.directory))
            self.root.after(0, self._update_ui_with_smart_results, found_rel_paths)
        except Exception as e:
            self._log_message(f"Smart Paster Error: {e}", "error")
        finally:
            self.root.after(0, lambda: self.smart_paste_text.config(state=tk.NORMAL))

    def _update_ui_with_smart_results(self, rel_paths: List[str]):
        if not rel_paths:
            self._log_message("Smart Paster: No files found.", "warning")
            return
        
        added_count = 0
        for rel_path in rel_paths:
            abs_path = os.path.join(self.directory, os.path.normpath(rel_path))
            if os.path.exists(abs_path):
                if self._add_file_to_selection(rel_path):
                    added_count += 1
        
        if added_count > 0:
            self._update_ui_state()
            self._log_message(f"Smart Paster: Added {added_count} new file(s) to selection.", "success")
        else:
            self._log_message("Smart Paster: All found files were already selected.", "info")

    def start_async_project_load(self):
        self._log_message("Scanning project files...")
        threading.Thread(target=self._project_load_worker, daemon=True).start()

    def _project_load_worker(self):
        self._scan_and_cache_all_files()
        self.root.after(0, self.finish_project_load)

    def finish_project_load(self):
        self.load_preset_into_ui()

    def _is_file_content_supported(self, filepath: str) -> bool:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in self.supported_binary_ext:
            return True
        return is_text_file(os.path.join(self.directory, filepath))

    def on_tree_double_click(self, event: tk.Event):
        item_id = self.tree.identify_row(event.y)
        if item_id:
            item = self.tree.item(item_id)
            if 'file' in item.get('tags', []) and item['values']:
                fp = item['values'][0]
                if self._add_file_to_selection(fp):
                    self._update_ui_state()
            elif 'folder' in item.get('tags', []):
                self.tree.selection_set(item_id)
                self.add_selected_folder()

    def repopulate_tree(self, files_to_display: Optional[List[str]] = None):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if files_to_display is None:
            self.tree.bind('<<TreeviewOpen>>', self.on_tree_expand)
            self.process_directory("", self.directory)
            return
        self.tree.unbind('<<TreeviewOpen>>')
        if not files_to_display:
            self.tree.insert("", "end", text="No matching files found.", tags=('info',))
            return
        nodes: Dict[str, str] = {"": ""}
        for file_path in sorted(files_to_display):
            parent_path = ""
            path_parts = file_path.split('/')
            for i, part in enumerate(path_parts[:-1]):
                current_path = os.path.join(parent_path, part)
                if current_path not in nodes:
                    nodes[current_path] = self.tree.insert(nodes.get(parent_path, ""), 'end', text=f"📁 {part}", values=[current_path.replace(os.path.sep, '/')], tags=('folder',), open=True)
                parent_path = current_path
            self.tree.insert(nodes.get(parent_path, ""), 'end', text=f"📄 {path_parts[-1]}", values=[file_path], tags=('file',))
        self.tree.tag_configure('file', foreground='#87CEEB')
        self.tree.tag_configure('folder', foreground='#DDA0DD')

    def add_all_visible_files(self):
        visible_files, added_count = [], 0
        def _collect_files(parent_id):
            for child_id in self.tree.get_children(parent_id):
                item = self.tree.item(child_id)
                if 'file' in item.get('tags', []) and item['values']:
                    visible_files.append(item['values'][0])
                elif 'folder' in item.get('tags', []):
                    _collect_files(child_id)
        _collect_files("")
        for fp in visible_files:
            if self._add_file_to_selection(fp):
                added_count += 1
        if added_count > 0:
            self._update_ui_state()
        self._log_message(f"Added {added_count} visible file(s).")

    def collapse_all_tree_items(self):
        for item in self.tree.get_children():
            self.tree.item(item, open=False)

    def load_preset_into_ui(self):
        name = self.preset_var.get()
        if not name or name not in self.presets:
            return
        self._log_message(f"Loading preset '{name}'...")
        data = self.presets[name]
        self.search_var.set(data.get("filter_text", ""))
        self.advanced_exclude_var.set(data.get("advanced_exclude_mode", False))
        self.exclude_dirs_var.set(data.get("exclude_dirs", " ".join(sorted(list(IGNORE_DIRS)))))
        self.exclude_patterns_var.set(data.get("exclude_patterns", "*.log *.json *.csv *.env .DS_Store .gitignore"))
        self.exclusion_var.set(data.get("exclusion_regex", r"venv/|\.git/|\.idea/|\.vscode/|__pycache__|/node_modules/|/build/|/dist/|.*\.log$"))
        
        saved_websocket_state = data.get("websocket_enabled", True)
        if saved_websocket_state != self.websocket_enabled:
            self.websocket_enabled = saved_websocket_state
            self.btn_toggle_server.config(text="Disable Server" if self.websocket_enabled else "Enable Server")
        self._toggle_exclude_mode()
        self._perform_filter(from_preset_load=True)
        self.clear_all(auto_save=False)
        added = 0
        for fp in data.get("selected_files", []):
            if os.path.exists(os.path.join(self.directory, os.path.normpath(fp))):
                if self._add_file_to_selection(fp):
                    added += 1
        self._update_ui_state(auto_save=False)
        self._log_message(f"Loaded preset '{name}'. ({added}/{len(data.get('selected_files', []))} files).")

    def auto_save_current_preset(self):
        name = self.preset_var.get()
        if not name:
            return
        data = {"selected_files": self._get_selected_files_ordered(), "filter_text": self.search_var.get(), "advanced_exclude_mode": self.advanced_exclude_var.get(), "exclude_dirs": self.exclude_dirs_var.get(), "exclude_patterns": self.exclude_patterns_var.get(), "exclusion_regex": self.exclusion_var.get(), "websocket_enabled": self.websocket_enabled}
        if self.presets.get(name) != data:
            self.presets[name] = data
            self.save_config()

    def _toggle_exclude_mode(self):
        if self.advanced_exclude_var.get():
            self.simple_exclude_frame.pack_forget()
            self.advanced_exclude_frame.pack(fill=tk.X, expand=True)
        else:
            self.advanced_exclude_frame.pack_forget()
            self.simple_exclude_frame.pack(fill=tk.X, expand=True)
        self._debounce_search()

    def _get_exclusion_regex(self) -> Optional[re.Pattern]:
        try:
            if self.advanced_exclude_var.get():
                if exclusion_str := self.exclusion_var.get():
                    return re.compile(exclusion_str, re.IGNORECASE)
                return None
            
            dirs = [d for d in self.exclude_dirs_var.get().split() if d]
            files = [p for p in self.exclude_patterns_var.get().split() if p]
            parts = []
            if dirs:
                sep = re.escape(os.path.sep)
                dir_alternations = "|".join(re.escape(d) for d in dirs)
                dir_pattern = f"(^|{sep})({dir_alternations})({sep}|$)"
                parts.append(dir_pattern)
            if files:
                file_patterns = [fnmatch.translate(p) for p in files]
                parts.extend(file_patterns)
            
            if not parts:
                return None
            
            final_regex_str = "|".join(f"({p})" for p in parts)
            return re.compile(final_regex_str, re.IGNORECASE)
        except re.error as e:
            self._log_message(f"Regex Error: {e}", 'error')
            return None

    def _scan_and_cache_all_files(self):
        all_files_list, regex = [], self._get_exclusion_regex()
        for root, dirs, files in os.walk(self.directory, topdown=True):
            rel_root = os.path.relpath(root, self.directory).replace(os.path.sep, '/')
            dirs[:] = [d for d in dirs if not (regex and regex.search(f"{rel_root}/{d}/".replace('./', '')))]
            for filename in files:
                rel_path = os.path.relpath(os.path.join(root, filename), self.directory).replace(os.path.sep, '/')
                if not (regex and regex.search(rel_path)) and is_includable_file(os.path.join(root, filename)):
                    all_files_list.append(rel_path)
        self.all_text_files = sorted(all_files_list, key=str.lower)

    def _perform_filter(self, from_preset_load: bool = False):
        search_term = self.search_var.get().lower()
        current_exclusion_state = (self.advanced_exclude_var.get(), self.exclude_dirs_var.get(), self.exclude_patterns_var.get(), self.exclusion_var.get())
        if not hasattr(self, '_last_exclusion_state') or self._last_exclusion_state != current_exclusion_state:
            self._last_exclusion_state = current_exclusion_state
            threading.Thread(target=self._scan_and_repopulate, args=(search_term, from_preset_load), daemon=True).start()
        else:
            files_to_display = [f for f in self.all_text_files if search_term in os.path.basename(f).lower()] if search_term else None
            self.repopulate_tree(files_to_display)
            if not from_preset_load:
                self._debounce_auto_save()

    def _scan_and_repopulate(self, search_term: str, from_preset_load: bool):
        self._scan_and_cache_all_files()
        def callback():
            files_to_display = [f for f in self.all_text_files if search_term in os.path.basename(f).lower()] if search_term else None
            self.repopulate_tree(files_to_display)
            if not from_preset_load:
                self._debounce_auto_save()
        self.root.after(0, callback)

    def _debounce_search(self, *args):
        if self._search_job:
            self.root.after_cancel(self._search_job)
        self._search_job = self.root.after(300, self._perform_filter)

    def load_project_config(self):
        try:
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    self.full_config = json.load(f)
        except (json.JSONDecodeError, IOError):
            self.full_config = {}
        if self.directory not in self.full_config:
            self.full_config[self.directory] = {"presets": {DEFAULT_PRESET_NAME: {}}, "last_active_preset": DEFAULT_PRESET_NAME}
        self.project_data = self.full_config[self.directory]
        self.presets = self.project_data.get('presets', {})
        if DEFAULT_PRESET_NAME not in self.presets:
            self.presets[DEFAULT_PRESET_NAME] = {}
        self.update_preset_combobox()
        self.preset_var.set(self.project_data.get("last_active_preset", DEFAULT_PRESET_NAME))

    def save_config(self, quiet: bool = True):
        self.project_data['last_active_preset'] = self.preset_var.get()
        self.project_data['presets'] = self.presets
        self.full_config[self.directory] = self.project_data
        try:
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.full_config, f, indent=4)
            if not quiet:
                self._log_message(f"Preset '{self.preset_var.get()}' saved.", 'success')
        except IOError as e:
            messagebox.showerror("Config Error", f"Could not save config: {e}")

    def _debounce_auto_save(self, *args):
        if self._auto_save_job:
            self.root.after_cancel(self._auto_save_job)
        self._auto_save_job = self.root.after(1500, self.auto_save_current_preset)

    def update_preset_combobox(self):
        self.preset_combobox['values'] = [DEFAULT_PRESET_NAME] + sorted([p for p in self.presets.keys() if p != DEFAULT_PRESET_NAME], key=str.lower)

    def save_current_as_preset(self):
        name = simpledialog.askstring("Save New Preset", "Enter a name:", parent=self.root)
        if not (name and name.strip()):
            return
        name = name.strip()
        if name in self.presets and not messagebox.askyesno("Confirm Overwrite", f"Preset '{name}' exists. Overwrite?", parent=self.root):
            return
        self.auto_save_current_preset()
        current_name = self.preset_var.get()
        if current_name in self.presets:
            self.presets[name] = self.presets[current_name]
        self.update_preset_combobox()
        self.preset_var.set(name)
        self.save_config(quiet=False)

    def on_preset_selected(self, event=None):
        self.load_preset_into_ui()
        self._debounce_auto_save()

    def remove_selected_preset(self):
        name = self.preset_var.get()
        if name == DEFAULT_PRESET_NAME:
            messagebox.showerror("Action Denied", "Default preset cannot be removed.")
            return
        if messagebox.askyesno("Confirm Deletion", f"Delete preset '{name}'?", parent=self.root):
            if name in self.presets:
                del self.presets[name]
                self.update_preset_combobox()
                self.preset_var.set(DEFAULT_PRESET_NAME)
                self.load_preset_into_ui()
                self.save_config()
                self._log_message(f"Preset '{name}' removed.", 'info')

    def _bind_select_all(self, w: tk.Widget):
        def sa(e=None):
            if isinstance(w, (ttk.Entry, tk.Entry)):
                w.select_range(0, 'end')
            elif isinstance(w, (scrolledtext.ScrolledText, tk.Text)):
                w.tag_add('sel', '1.0', 'end')
            return "break"
        w.bind("<Control-a>", sa)
        w.bind("<Command-a>", sa)

    def _setup_interrupt_handler(self):
        self.interrupted = False
        try:
            signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'interrupted', True))
        except (ValueError, TypeError):
            pass
        self.root.after(250, self._check_for_interrupt)

    def _check_for_interrupt(self):
        if self.interrupted:
            self.on_closing()
        else:
            self.root.after(250, self._check_for_interrupt)

    def process_directory(self, parent_id: str, path: str):
        try:
            items = sorted(os.listdir(path), key=str.lower)
        except (OSError, PermissionError):
            return
        regex = self._get_exclusion_regex()
        for cid in self.tree.get_children(parent_id):
            if self.tree.item(cid, "values") == ("dummy",):
                self.tree.delete(cid)
        for name in items:
            full_path = os.path.join(path, name)
            rel_path = os.path.relpath(full_path, self.directory).replace(os.path.sep, '/')
            is_dir = os.path.isdir(full_path)
            check_path = rel_path + '/' if is_dir else rel_path
            if regex and regex.search(check_path):
                continue
            if is_dir:
                did = self.tree.insert(parent_id, 'end', text=f"📁 {name}", values=[rel_path], tags=('folder',))
                self.tree.insert(did, 'end', text='...', values=['dummy'])
            elif is_includable_file(full_path):
                self.tree.insert(parent_id, 'end', text=f"📄 {name}", values=[rel_path], tags=('file',))
        self.tree.tag_configure('file', foreground='#87CEEB')
        self.tree.tag_configure('folder', foreground='#DDA0DD')

    def on_tree_expand(self, event: Optional[tk.Event]):
        item_id = self.tree.focus()
        self._populate_tree_node(item_id)
        
    def _populate_tree_node(self, item_id: str):
        if not item_id or not (children := self.tree.get_children(item_id)) or self.tree.item(children[0], "values") != ("dummy",):
            return
        if full_path_parts := self.tree.item(item_id, "values"):
            self.process_directory(item_id, os.path.join(self.directory, full_path_parts[0].replace('/', os.path.sep)))
            
    def expand_all_tree_items(self):
        for item in self.tree.get_children():
            self._expand_tree_item_recursive(item)
            
    def _expand_tree_item_recursive(self, item_id: str):
        self._populate_tree_node(item_id)
        if self.tree.get_children(item_id):
            self.tree.item(item_id, open=True)
            for child in self.tree.get_children(item_id):
                self._expand_tree_item_recursive(child)

    def get_all_files_in_folder(self, path: str) -> List[str]:
        return [f for f in self.all_text_files if f.replace(os.path.sep, '/').startswith(path.replace(os.path.sep, '/') + '/')]

    def add_selected_folder(self):
        if not self.tree.selection():
            return
        item = self.tree.item(self.tree.selection()[0])
        if 'folder' not in item['tags']:
            return
        path, files, count = item['values'][0], self.get_all_files_in_folder(item['values'][0]), 0
        for fp in files:
            if self._add_file_to_selection(fp):
                count += 1
        if count > 0:
            self._update_ui_state()
        self._log_message(f"Added {count} file(s) from {os.path.basename(path)}.")

    def select_all_files(self, event=None):
        """Select all files in the selected files tree."""
        all_items = self.selected_files_tree.get_children()
        if all_items:
            self.selected_files_tree.selection_set(all_items)
        return "break"  # Prevent further event propagation

    def _on_column_click(self, column: str):
        """Handle column header click to sort files."""
        # Toggle sort direction if same column, otherwise default to ascending
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False

        # Disable drag and drop while sorted
        self.drag_enabled = False

        # Perform the sort
        self._sort_selected_files()

        # Update header text to show sort indicator
        self._update_column_headers()

    def _update_column_headers(self):
        """Update column headers to show sort indicators."""
        headers = {
            "filepath": "File Path",
            "filetype": "Type",
            "filesize": "Size",
            "char_count": "Characters",
            "changed": "Modified"
        }

        for col, text in headers.items():
            if col == self.sort_column:
                indicator = " ▼" if self.sort_reverse else " ▲"
                self.selected_files_tree.heading(col, text=text + indicator)
            else:
                self.selected_files_tree.heading(col, text=text)

    def _sort_selected_files(self):
        """Sort files in the tree by the current sort column."""
        if not self.sort_column:
            return

        # Gather all items with their data
        items = []
        for item_id in self.selected_files_tree.get_children():
            values = self.selected_files_tree.item(item_id, "values")
            tags = self.selected_files_tree.item(item_id, "tags")
            items.append((values, tags))

        # Define sort key based on column
        def get_sort_key(item):
            values = item[0]
            filepath = values[0]  # First column is always filepath

            col_index = {
                "filepath": 0,
                "filetype": 1,
                "filesize": 2,
                "char_count": 3,
                "changed": 4
            }[self.sort_column]

            value = values[col_index] if col_index < len(values) else ""

            # Handle special cases for numeric columns using stored metadata
            if self.sort_column == "filesize":
                # Use raw bytes from metadata for proper sorting
                metadata = self.file_metadata.get(filepath, {})
                return metadata.get('filesize_bytes', 0)

            elif self.sort_column == "char_count":
                # Handle "N/A" and numeric values
                try:
                    return int(value.replace(",", "")) if value != "N/A" else -1
                except (ValueError, AttributeError):
                    return -1

            elif self.sort_column == "changed":
                # Use raw timestamp from metadata for proper sorting
                metadata = self.file_metadata.get(filepath, {})
                return metadata.get('timestamp', 0)

            # For text columns, use natural string sorting
            return str(value).lower()

        # Sort items
        items.sort(key=get_sort_key, reverse=self.sort_reverse)

        # Clear tree and repopulate in sorted order
        for item_id in self.selected_files_tree.get_children():
            self.selected_files_tree.delete(item_id)

        for values, tags in items:
            self.selected_files_tree.insert("", tk.END, values=values, tags=tags)

        self._update_ui_state()

    def remove_selected(self):
        selection = self.selected_files_tree.selection()
        if selection:
            # Remove all selected items
            for selected_item in selection:
                fp = self.selected_files_tree.item(selected_item, "values")[0]
                self.selected_files_tree.delete(selected_item)
                if fp in self.selected_files_map:
                    del self.selected_files_map[fp]
                if fp in self.file_metadata:
                    del self.file_metadata[fp]
            self._update_ui_state()

    def clear_all(self, auto_save: bool = True):
        for item in self.selected_files_tree.get_children():
            self.selected_files_tree.delete(item)
        self.selected_files_map.clear()
        self.file_metadata.clear()
        self._update_ui_state(auto_save=auto_save)

    def update_selected_count(self):
        c = len(self.selected_files_tree.get_children())
        self.selected_count_var.set(f"{c} file{'s' if c != 1 else ''} selected")

    def toggle_preview(self):
        self.preview_visible = not self.preview_visible
        if self.preview_visible:
            self.preview_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 0), in_=self.main_container)
            self.update_preview()
        else:
            self.preview_frame.pack_forget()
        self.btn_toggle_preview.configure(text="Hide Preview" if self.preview_visible else "Show Preview")

    def copy_to_clipboard(self):
        if pyperclip is None:
            messagebox.showerror("Error", "Install pyperclip: pip install pyperclip")
            return
        selected = self._get_selected_files_ordered()
        if not selected:
            self._log_message("Copy: No files selected.", 'warning')
            return
        self._log_message("Copy: Processing files...")
        self.root.update_idletasks()
        out = build_clipboard_content([os.path.join(self.directory, f) for f in selected], self.directory)
        pyperclip.copy(out)
        size_kb = len(out) / 1024
        self._log_message(f"Copied {len(selected)} file(s) to clipboard! ({size_kb:.1f} KB)", 'success')

    def on_drag_start(self, event: tk.Event):
        if not self.drag_enabled:
            return
        self.drag_start_item = self.selected_files_tree.identify_row(event.y)

    def on_drag_motion(self, event: tk.Event):
        if not self.drag_enabled or not self.drag_start_item:
            return

        item_over = self.selected_files_tree.identify_row(event.y)
        if item_over and item_over != self.drag_start_item:
            # Check if drag_start_item still exists in the tree
            if self.selected_files_tree.exists(self.drag_start_item):
                self.selected_files_tree.move(self.drag_start_item, "", self.selected_files_tree.index(item_over))
                self._update_ui_state()
            else:
                # Reset drag operation if item no longer exists
                self.drag_start_item = None
    
    def _add_file_to_selection(self, filepath: str) -> bool:
        if filepath in self.selected_files_map:
            return False

        self.selected_files_map[filepath] = True

        # Gather file metadata
        full_path = os.path.join(self.directory, filepath)

        # Get file type (extension)
        filetype = os.path.splitext(filepath)[1] or "N/A"

        # Get file size and timestamp
        try:
            file_stat = os.stat(full_path)
            filesize_bytes = file_stat.st_size
            filesize_str = format_filesize(filesize_bytes)
            timestamp = file_stat.st_mtime

            # Get last modified time
            changed_str = format_timestamp(timestamp)

            # Store raw metadata for sorting
            self.file_metadata[filepath] = {
                'filesize_bytes': filesize_bytes,
                'timestamp': timestamp
            }
        except (OSError, FileNotFoundError):
            filesize_str = "N/A"
            changed_str = "N/A"
            self.file_metadata[filepath] = {
                'filesize_bytes': 0,
                'timestamp': 0
            }

        # Get character count (existing logic)
        char_count_str = "N/A"
        try:
            if is_text_file(full_path):
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    char_count_str = f"{len(f.read()):,}"
        except Exception:
            pass # Keep N/A

        tags = () if self._is_file_content_supported(filepath) else ('unsupported',)
        self.selected_files_tree.insert("", tk.END, values=(filepath, filetype, filesize_str, char_count_str, changed_str), tags=tags)
        return True

    # --- WebSocket Server Control Methods ---
    def toggle_websocket_server(self):
        if self.websocket_enabled:
            self._stop_websocket_server()
            self.websocket_enabled = False
            self.btn_toggle_server.config(text="Enable Server")
            self._log_message("WebSocket server disabled.", "info")
        else:
            self.websocket_enabled = True
            self.btn_toggle_server.config(text="Disable Server")
            threading.Thread(target=self._start_websocket_server, daemon=True).start()
            self._log_message("WebSocket server enabled.", "info")
    
    def restart_websocket_server(self):
        self._log_message("Restarting WebSocket server...", "info")
        self._stop_websocket_server()
        if self.websocket_enabled:
            threading.Thread(target=self._start_websocket_server, daemon=True).start()
    
    def _stop_websocket_server(self):
        try:
            if self.websocket_server:
                self.websocket_server.close()
                if (self.websocket_loop and not self.websocket_loop.is_closed() and self.websocket_loop.is_running()):
                    def close_server():
                        try:
                            if self.websocket_server:
                                asyncio.create_task(self.websocket_server.wait_closed())
                        except Exception:
                            pass
                    self.websocket_loop.call_soon_threadsafe(close_server)
        except Exception:
            pass
        finally:
            self._cleanup_websocket_state()
    
    def disconnect_all_clients(self):
        if not self.connected_clients:
            self._log_message("No clients to disconnect.", "info")
            return
        
        client_count = len(self.connected_clients)
        clients_to_close = list(self.connected_clients)
        for client in clients_to_close:
            if (self.websocket_loop and not self.websocket_loop.is_closed() and self.websocket_loop.is_running()):
                try:
                    asyncio.run_coroutine_threadsafe(client.close(), self.websocket_loop)
                except Exception:
                    pass
        self._log_message(f"Disconnected {client_count} client(s).", "info")
    
    def _refresh_connections_display(self):
        server_running = (self.websocket_server and self.websocket_loop and self.websocket_loop.is_running())
        
        if server_running:
            self.server_status_var.set("Running")
            self.status_label.config(foreground="#4CAF50")
            if self.websocket_start_time:
                uptime = datetime.now() - self.websocket_start_time
                hours, remainder = divmod(int(uptime.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                self.server_uptime_var.set(f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        elif not self.websocket_enabled:
            self.server_status_var.set("Disabled")
            self.status_label.config(foreground="#FFC107")
            self.server_uptime_var.set("Uptime: --")
        else:
            self.server_status_var.set("Stopped")
            self.status_label.config(foreground="#F44336")
            self.server_uptime_var.set("Uptime: --")
        
        if self.actual_websocket_port:
            port_text = f"Port: {self.actual_websocket_port}"
            if self.actual_websocket_port != self.websocket_port:
                port_text += f" (configured: {self.websocket_port})"
            self.server_port_var.set(port_text)
        else:
            self.server_port_var.set(f"Port: {self.websocket_port} (configured)")
        
        self.client_count_var.set(f"Connected: {len(self.connected_clients)}")
        
        self.clients_listbox.delete(0, tk.END)
        if self.client_info:
            for websocket, info in self.client_info.items():
                duration_str = f"{int((datetime.now() - info['connect_time']).total_seconds())}s"
                self.clients_listbox.insert(tk.END, f"{info['address']} - Connected: {duration_str}")
        
        if self.connections_refresh_job:
            self.root.after_cancel(self.connections_refresh_job)
        self.connections_refresh_job = self.root.after(1000, self._refresh_connections_display)
    
    def _cleanup_websocket_state(self):
        self.websocket_server = None
        self.websocket_loop = None
        self.websocket_start_time = None
        self.actual_websocket_port = None
        self.connected_clients.clear()
        self.client_info.clear()
    
    def _save_websocket_config(self):
        if not self.actual_websocket_port:
            return
        try:
            websocket_config = {
                'host': self.websocket_host,
                'port': self.actual_websocket_port,
                'start_time': self.websocket_start_time.isoformat() if self.websocket_start_time else None,
                'uri': f"ws://{self.websocket_host}:{self.actual_websocket_port}"
            }
            if 'global' not in self.full_config:
                self.full_config['global'] = {}
            self.full_config['global']['websocket_server'] = websocket_config
            self.save_config(quiet=True)
            self._log_message("WebSocket server config updated", "info")
        except Exception as e:
            self._log_message(f"Failed to save WebSocket config: {e}", "warning")
    
    def _safe_broadcast_update(self):
        if not self.websocket_server or not self.connected_clients or not self.websocket_loop:
            return
        try:
            if not self.websocket_loop.is_closed() and self.websocket_loop.is_running():
                asyncio.run_coroutine_threadsafe(self._broadcast_update(), self.websocket_loop)
        except Exception:
            pass
    
    def _delayed_websocket_start(self):
        if self.websocket_enabled:
            threading.Thread(target=self._start_websocket_server, daemon=True).start()
            self.root.after(2000, self._safe_broadcast_update)


class PreviewChangesDialog:
    """Modal dialog for previewing file changes with GitHub-style diff display."""
    
    def __init__(self, parent: tk.Tk, changes: List[FileChange], directory: str):
        self.parent = parent
        self.changes = changes
        self.directory = directory
        self.result = None  # Will be set to 'apply', 'apply_selected', or None
        
        # Create modal window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Preview File Changes")
        self.dialog.geometry("1200x800")
        self.dialog.configure(bg=DARK_BG)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (800 // 2)
        self.dialog.geometry(f"1200x800+{x}+{y}")
        
        self._setup_styles()
        self._create_widgets()
        self._populate_data()
        
        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def _setup_styles(self):
        """Apply dark theme styles to the dialog."""
        style = ttk.Style()
        
        # Configure styles for the dialog
        style.configure("Preview.TFrame", background=DARK_BG)
        style.configure("Preview.TLabel", background=DARK_BG, foreground=DARK_FG)
        style.configure("Preview.TButton", background=DARK_BUTTON_BG, foreground=DARK_FG)
        style.configure("Preview.Treeview", background=DARK_TREE_BG, foreground=DARK_FG, fieldbackground=DARK_TREE_BG)
        style.map("Preview.Treeview", background=[('selected', DARK_SELECT_BG)])
        
    def _create_widgets(self):
        """Create and layout all dialog widgets."""
        main_frame = ttk.Frame(self.dialog, style="Preview.TFrame", padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with summary
        header_frame = ttk.Frame(main_frame, style="Preview.TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text="File Changes Preview", 
                 font=("Segoe UI", 14, "bold"), style="Preview.TLabel").pack(side=tk.LEFT)
        
        self.summary_label = ttk.Label(header_frame, style="Preview.TLabel")
        self.summary_label.pack(side=tk.RIGHT)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Files overview tab
        self._create_files_tab()
        
        # Individual file tabs
        self._create_file_tabs()
        
        # Bottom buttons
        buttons_frame = ttk.Frame(main_frame, style="Preview.TFrame")
        buttons_frame.pack(fill=tk.X)
        
        ttk.Button(buttons_frame, text="Cancel", command=self._on_cancel,
                  style="Preview.TButton").pack(side=tk.LEFT)
        
        ttk.Button(buttons_frame, text="Apply Selected", command=self._on_apply_selected,
                  style="Preview.TButton").pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(buttons_frame, text="Apply All", command=self._on_apply_all,
                  style="Preview.TButton").pack(side=tk.RIGHT)
    
    def _create_files_tab(self):
        """Create the files overview tab."""
        files_frame = ttk.Frame(self.notebook, style="Preview.TFrame", padding=10)
        self.notebook.add(files_frame, text="Files Overview")
        
        # Files tree
        tree_frame = ttk.Frame(files_frame, style="Preview.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.files_tree = ttk.Treeview(tree_frame, 
                                      columns=("status", "type", "changes"), 
                                      show="tree headings",
                                      style="Preview.Treeview")
        
        self.files_tree.heading("#0", text="File Path", anchor='w')
        self.files_tree.heading("status", text="Status", anchor='w')
        self.files_tree.heading("type", text="Type", anchor='w')
        self.files_tree.heading("changes", text="Changes", anchor='w')
        
        self.files_tree.column("#0", width=400, stretch=tk.YES)
        self.files_tree.column("status", width=150, stretch=tk.NO)
        self.files_tree.column("type", width=100, stretch=tk.NO)
        self.files_tree.column("changes", width=150, stretch=tk.NO)
        
        # Add checkboxes functionality
        self.files_tree.bind("<Button-1>", self._on_tree_click)
        
        # Scrollbar for tree
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.files_tree.yview)
        self.files_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.files_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Selection controls
        controls_frame = ttk.Frame(files_frame, style="Preview.TFrame")
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(controls_frame, text="Select All", command=self._select_all,
                  style="Preview.TButton").pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Select None", command=self._select_none,
                  style="Preview.TButton").pack(side=tk.LEFT, padx=(5, 0))
        
    def _create_file_tabs(self):
        """Create individual tabs for each file with diff view."""
        for i, change in enumerate(self.changes):
            if change.change_type == ChangeType.INVALID_PATH:
                continue
                
            tab_frame = ttk.Frame(self.notebook, style="Preview.TFrame", padding=10)
            tab_name = os.path.basename(change.file_path)
            if len(tab_name) > 20:
                tab_name = tab_name[:17] + "..."
            self.notebook.add(tab_frame, text=tab_name)
            
            # File info header
            info_frame = ttk.Frame(tab_frame, style="Preview.TFrame")
            info_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(info_frame, text=f"File: {change.file_path}", 
                     font=("Segoe UI", 11, "bold"), style="Preview.TLabel").pack(anchor='w')
            ttk.Label(info_frame, text=change.status_summary, 
                     style="Preview.TLabel").pack(anchor='w')
            
            # Diff content
            diff_text = scrolledtext.ScrolledText(tab_frame, 
                                                 wrap=tk.NONE, 
                                                 bg=DARK_ENTRY_BG, 
                                                 fg=DARK_FG, 
                                                 font=("Consolas", 10),
                                                 borderwidth=0, 
                                                 highlightthickness=1)
            diff_text.pack(fill=tk.BOTH, expand=True)
            
            # Configure diff highlighting
            diff_text.tag_config("added", foreground="#22C55E", background="#0F2A1A")
            diff_text.tag_config("removed", foreground="#F87171", background="#2A0F0F") 
            diff_text.tag_config("context", foreground="#9CA3AF")
            diff_text.tag_config("header", foreground="#60A5FA", font=("Consolas", 10, "bold"))
            
            self._populate_diff_content(diff_text, change)
            diff_text.config(state=tk.DISABLED)
    
    def _populate_diff_content(self, text_widget: scrolledtext.ScrolledText, change: FileChange):
        """Populate the diff content for a file change."""
        if change.change_type == ChangeType.NEW_FILE:
            text_widget.insert(tk.END, f"New file: {change.file_path}\n", "header")
            text_widget.insert(tk.END, f"+++ {change.file_path}\n", "header")
            for line_num, line in enumerate(change.content.splitlines(), 1):
                text_widget.insert(tk.END, f"+{line_num:4d}: {line}\n", "added")
        elif change.change_type == ChangeType.MODIFY_FILE and change.diff_lines:
            for line in change.diff_lines:
                if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
                    text_widget.insert(tk.END, line + '\n', "header")
                elif line.startswith('+'):
                    text_widget.insert(tk.END, line + '\n', "added")
                elif line.startswith('-'):
                    text_widget.insert(tk.END, line + '\n', "removed")
                else:
                    text_widget.insert(tk.END, line + '\n', "context")
    
    def _populate_data(self):
        """Populate the files tree with change data."""
        total_files = len(self.changes)
        new_files = sum(1 for c in self.changes if c.change_type == ChangeType.NEW_FILE)
        modified_files = sum(1 for c in self.changes if c.change_type == ChangeType.MODIFY_FILE)
        invalid_files = sum(1 for c in self.changes if c.change_type == ChangeType.INVALID_PATH)
        
        self.summary_label.config(text=f"{total_files} files ({new_files} new, {modified_files} modified, {invalid_files} invalid)")
        
        for i, change in enumerate(self.changes):
            checkbox_text = "☑" if change.selected else "☐"
            if change.change_type == ChangeType.INVALID_PATH:
                status = "❌ Invalid"
                type_text = "Error"
                checkbox_text = "☐"  # Invalid files can't be selected
            elif change.change_type == ChangeType.NEW_FILE:
                status = "📄 New File"
                type_text = "New"
            else:
                status = "📝 Modified"
                type_text = "Modified"
            
            item_id = self.files_tree.insert("", tk.END, 
                                           text=f"{checkbox_text} {change.file_path}",
                                           values=(status, type_text, change.status_summary),
                                           tags=(f"change_{i}",))
    
    def _on_tree_click(self, event):
        """Handle clicking on tree items to toggle selection."""
        item = self.files_tree.identify('item', event.x, event.y)
        if item:
            tags = self.files_tree.item(item, 'tags')
            if tags:
                change_idx = int(tags[0].split('_')[1])
                change = self.changes[change_idx]
                
                if change.change_type != ChangeType.INVALID_PATH:
                    change.selected = not change.selected
                    checkbox_text = "☑" if change.selected else "☐"
                    current_text = self.files_tree.item(item, 'text')
                    new_text = f"{checkbox_text} {change.file_path}"
                    self.files_tree.item(item, text=new_text)
    
    def _select_all(self):
        """Select all valid changes."""
        for i, change in enumerate(self.changes):
            if change.change_type != ChangeType.INVALID_PATH:
                change.selected = True
        self._refresh_tree()
    
    def _select_none(self):
        """Deselect all changes."""
        for change in self.changes:
            change.selected = False
        self._refresh_tree()
    
    def _refresh_tree(self):
        """Refresh the tree display."""
        for item in self.files_tree.get_children():
            self.files_tree.delete(item)
        self._populate_data()
    
    def _on_apply_all(self):
        """Apply all valid changes."""
        for change in self.changes:
            if change.change_type != ChangeType.INVALID_PATH:
                change.selected = True
        self.result = 'apply_selected'
        self.dialog.destroy()
    
    def _on_apply_selected(self):
        """Apply only selected changes."""
        self.result = 'apply_selected'
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Cancel the dialog."""
        self.result = None
        self.dialog.destroy()