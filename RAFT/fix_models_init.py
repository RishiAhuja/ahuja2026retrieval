#!/usr/bin/env python3
"""
FIX SCRIPT: Repair models/__init__.py to correctly import PatchTST
"""

import os

# Fix the models/__init__.py file
init_file = './models/__init__.py'

correct_content = """from .RAFT import Model as RAFT
from .TransformerLongContext import Model as TransformerLongContext
from .PatchTST import Model as PatchTST

__all__ = ['RAFT', 'TransformerLongContext', 'PatchTST']
"""

print("=" * 80)
print("FIXING models/__init__.py")
print("=" * 80)

# Backup current file
if os.path.exists(init_file):
    with open(init_file, 'r') as f:
        old_content = f.read()
    
    print("\nCurrent content:")
    print("-" * 80)
    print(old_content)
    print("-" * 80)
    
    # Write backup
    backup_file = init_file + '.backup'
    with open(backup_file, 'w') as f:
        f.write(old_content)
    print(f"\n✓ Backup saved to: {backup_file}")

# Write correct content
with open(init_file, 'w') as f:
    f.write(correct_content)

print(f"\n✓ Fixed {init_file}")
print("\nNew content:")
print("-" * 80)
print(correct_content)
print("-" * 80)

# Verify by importing
print("\nVerifying fix...")
try:
    # Force reimport
    import sys
    if 'models' in sys.modules:
        del sys.modules['models']
    if 'models.RAFT' in sys.modules:
        del sys.modules['models.RAFT']
    if 'models.TransformerLongContext' in sys.modules:
        del sys.modules['models.TransformerLongContext']
    if 'models.PatchTST' in sys.modules:
        del sys.modules['models.PatchTST']
    
    from models import RAFT, TransformerLongContext, PatchTST
    
    print(f"  ✓ RAFT type: {type(RAFT)}")
    print(f"  ✓ TransformerLongContext type: {type(TransformerLongContext)}")
    print(f"  ✓ PatchTST type: {type(PatchTST)}")
    
    print(f"\n  ✓ callable(RAFT): {callable(RAFT)}")
    print(f"  ✓ callable(TransformerLongContext): {callable(TransformerLongContext)}")
    print(f"  ✓ callable(PatchTST): {callable(PatchTST)}")
    
    if callable(PatchTST):
        print("\n✅ SUCCESS! PatchTST is now a callable class!")
    else:
        print("\n❌ FAILED! PatchTST is still not callable!")
        
except Exception as e:
    print(f"\n❌ Verification failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("FIX COMPLETE - Run validate_etth2.py again")
print("=" * 80)
