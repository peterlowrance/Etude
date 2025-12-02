#!/usr/bin/env python3
"""
Patch madmom to work with Python 3.12+
This fixes the MutableSequence import issue.
"""
import sys
from pathlib import Path

def fix_madmom():
    """Patch madmom's processors.py to use collections.abc instead of collections"""
    try:
        import madmom
        madmom_path = Path(madmom.__file__).parent
        processors_file = madmom_path / "processors.py"
        
        if not processors_file.exists():
            print(f"Could not find {processors_file}")
            return False
            
        content = processors_file.read_text()
        
        # Check if already patched
        if "from collections.abc import MutableSequence" in content:
            print("madmom is already patched")
            return True
            
        # Apply patch
        old_import = "from collections import MutableSequence"
        new_import = "from collections.abc import MutableSequence"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            processors_file.write_text(content)
            print("Successfully patched madmom!")
            return True
        else:
            print("Could not find the import statement to patch")
            return False
            
    except Exception as e:
        print(f"Error patching madmom: {e}")
        return False

if __name__ == "__main__":
    success = fix_madmom()
    sys.exit(0 if success else 1)
