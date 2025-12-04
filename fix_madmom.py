#!/usr/bin/env python3
"""
Patch madmom to work with Python 3.12+ and NumPy 1.24+
This fixes:
1. MutableSequence import issue (Python 3.10+)
2. np.float deprecation issue (NumPy 1.24+)
"""
import sys
import site
import os
from pathlib import Path

def find_madmom():
    print("Searching for madmom installation...")
    
    # Get all site-packages directories
    site_packages = site.getsitepackages()
    site_packages.append(site.getusersitepackages())
    
    madmom_path = None
    for sp in site_packages:
        candidate = Path(sp) / "madmom"
        if candidate.exists() and candidate.is_dir():
            madmom_path = candidate
            print(f"Found madmom at: {madmom_path}")
            return madmom_path
    
    # Fallback: try to find it via pip output if possible, or just standard paths
    print("Could not find madmom in standard site-packages. Checking common paths...")
    common_paths = [
        Path("/usr/local/lib/python3.10/dist-packages/madmom"),
        Path("/usr/local/lib/python3.11/dist-packages/madmom"),
        Path("/usr/local/lib/python3.12/dist-packages/madmom"),
    ]
    for p in common_paths:
        if p.exists():
            madmom_path = p
            print(f"Found madmom at: {madmom_path}")
            return madmom_path
            
    return None

def fix_mutable_sequence(madmom_path):
    print("\n[1/2] Fixing MutableSequence import...")
    processors_file = madmom_path / "processors.py"
    if not processors_file.exists():
        print(f"[ERROR] Could not find {processors_file}")
        return False
        
    try:
        content = processors_file.read_text()
        
        if "from collections.abc import MutableSequence" in content:
            print(" - Already patched.")
            return True
            
        old_import = "from collections import MutableSequence"
        new_import = "from collections.abc import MutableSequence"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            processors_file.write_text(content)
            print(" - Successfully patched!")
            return True
        else:
            print(" - Could not find the import statement to patch.")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to patch file: {e}")
        return False

def fix_numpy_float(madmom_path):
    print("\n[2/2] Fixing np.float deprecation...")
    # The error was in madmom/io/__init__.py
    # We will recursively search for np.float usage in all .py files to be safe
    
    patched_count = 0
    
    for root, dirs, files in os.walk(madmom_path):
        for file in files:
            if file.endswith(".py") or file.endswith(".pyx"):
                file_path = Path(root) / file
                try:
                    content = file_path.read_text()
                    if "np.float" in content:
                        # Replace np.float with float
                        # We use a simple replace here. 
                        # Note: np.float32/64 are fine, but np.float is not.
                        # Replacing "np.float" with "float" might break "np.float32" -> "float32" which is valid python but maybe not what we want if it was "np.float32".
                        # But "np.float" usually appears as "np.float)" or "np.float," or "dtype=np.float".
                        # To be safe, let's replace "np.float " and "np.float," and "np.float)" and "np.float]"
                        
                        # Actually, "float32" is not available in global scope usually, so "np.float32" -> "float32" would break if we just did string replace of "np.float".
                        # So we must be careful.
                        
                        # The specific error is `np.float`.
                        # Let's use a regex or just be specific about the replacement.
                        # However, for this specific error in io/__init__.py, it is `np.float`.
                        
                        # Let's just fix the reported file for now to avoid side effects, 
                        # or be very specific with replacement.
                        
                        # Strategy: Replace "np.float," with "float," etc.
                        # We need to handle np.float, np.int, np.bool, np.object, np.str
                        
                        new_content = content
                        # Map of deprecated numpy types to python builtins
                        type_map = {
                            "np.float": "float",
                            "np.int": "int",
                            "np.bool": "bool",
                            "np.object": "object",
                            "np.str": "str",
                        }
                        
                        suffixes = [",", ")", "]", " "]
                        
                        changed = False
                        for np_type, py_type in type_map.items():
                            for suffix in suffixes:
                                search_str = np_type + suffix
                                if search_str in new_content:
                                    new_content = new_content.replace(search_str, py_type + suffix)
                                    changed = True
                        
                        if changed:
                            file_path.write_text(new_content)
                            print(f" - Patched {file_path.name}")
                            patched_count += 1
                            
                except Exception as e:
                    print(f" - Failed to read/patch {file_path}: {e}")

    if patched_count == 0:
        print(" - No files needed patching (or none found).")
    else:
        print(f" - Successfully patched {patched_count} files.")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Patch madmom for Python 3.12+ and NumPy 1.24+")
    parser.add_argument("--path", type=str, help="Path to madmom source or installation directory", default=None)
    args = parser.parse_args()

    if args.path:
        madmom_path = Path(args.path)
        if not madmom_path.exists():
             print(f"[ERROR] Provided path does not exist: {madmom_path}")
             return False
        print(f"Patching madmom at provided path: {madmom_path}")
    else:
        madmom_path = find_madmom()
        
    if not madmom_path:
        print("[ERROR] Could not locate madmom installation directory.")
        return False

    success1 = fix_mutable_sequence(madmom_path)
    success2 = fix_numpy_float(madmom_path)
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
