import os
import sys
from typing import Optional, Any


def find_dia_locations() -> list:
    """Find possible locations of the Dia module."""
    import glob
    
    # List of potential locations to check
    potential_paths = [
        os.path.expanduser("~/.local/lib/python*/site-packages/dia"),
        os.path.expanduser("~/dia"),
        os.path.expanduser("~/dia_temp"),
        os.path.expanduser("~/repos/*/dia"),
        os.path.expanduser("~/repos/dia"),
        "/usr/local/lib/python*/site-packages/dia",
        "/usr/lib/python*/site-packages/dia",
        os.path.expanduser("~/miniconda3/lib/python*/site-packages/dia"),
        os.path.expanduser("~/anaconda3/lib/python*/site-packages/dia"),
        "dia",
        "dia_temp",
        "dia_temp/dia",
    ]
    
    # Check the current directory and all directories up to the root
    current_dir = os.getcwd()
    while current_dir != os.path.dirname(current_dir):  # Stop at root
        potential_paths.append(os.path.join(current_dir, "dia"))
        potential_paths.append(os.path.join(current_dir, "dia_temp/dia"))
        current_dir = os.path.dirname(current_dir)
    
    found_locations = []
    for pattern in potential_paths:
        matches = glob.glob(pattern)
        for match in matches:
            if os.path.isdir(match) and os.path.exists(os.path.join(match, "__init__.py")):
                found_locations.append(match)
    
    return found_locations


def add_dia_to_path() -> bool:
    """
    Attempt to find and add Dia to the Python path.
    Returns True if successful, False otherwise.
    """
    # First check if dia is already importable
    try:
        import dia
        return True
    except ImportError:
        pass
    
    # Try to find dia
    locations = find_dia_locations()
    if not locations:
        print("Could not find Dia installation")
        return False
    
    # Add the first found location to path
    dia_location = os.path.dirname(locations[0])  # Get the parent directory
    print(f"Adding Dia location to path: {dia_location}")
    sys.path.insert(0, dia_location)
    
    # Verify it worked
    try:
        import dia
        print(f"Successfully added Dia to path: {dia.__file__}")
        return True
    except ImportError:
        print("Failed to import Dia even after adding to path")
        return False


def get_dia_model(force_cpu: bool = False, **kwargs) -> Optional[Any]:
    """
    Get a Dia model instance, handling errors and path issues.
    
    Args:
        force_cpu: Whether to force CPU usage even if GPU is available
        **kwargs: Additional arguments to pass to Dia.from_pretrained
    
    Returns:
        The model or None if it could not be loaded.
    """
    # Make sure dia is in the path
    if not add_dia_to_path():
        print("ERROR: Could not add Dia to Python path")
        return None
    
    # Now try to import and create the model
    try:
        from dia.model import Dia
        
        # Set device to CPU if forced or as fallback
        if force_cpu:
            print("Forcing CPU usage for Dia model (may be very slow)")
            if "device" not in kwargs:
                kwargs["device"] = "cpu"
        
        print(f"Initializing Dia model on {'CPU' if force_cpu or kwargs.get('device') == 'cpu' else 'GPU'}...")
        
        # Use the from_pretrained method instead of direct initialization
        # Default to using nari-labs/Dia-1.6B if no model_id is provided
        model_id = kwargs.pop("model_id", "nari-labs/Dia-1.6B")
        
        try:
            model = Dia.from_pretrained(model_id, **kwargs)
            return model
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and not force_cpu:
                print("CUDA out of memory error. Trying to fall back to CPU (will be very slow)...")
                # Try again with CPU
                kwargs["device"] = "cpu"
                return get_dia_model(force_cpu=True, model_id=model_id, **kwargs)
            else:
                raise  # Re-raise other runtime errors or if CPU fallback was already attempted
    except ImportError as e:
        print(f"ERROR importing Dia model: {str(e)}")
        return None
    except Exception as e:
        print(f"ERROR creating Dia model: {str(e)}")
        return None
