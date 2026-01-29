#!/usr/bin/env python3
"""Debug script to check model imports"""

import sys
import os

# Add RAFT root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
raft_root = os.path.dirname(script_dir)
if raft_root not in sys.path:
    sys.path.insert(0, raft_root)

print("=" * 80)
print("DEBUGGING MODEL IMPORTS")
print("=" * 80)
print(f"Python path: {sys.path[0]}")
print(f"RAFT root: {raft_root}")
print()

# Test imports
print("1. Testing models import...")
try:
    import models
    print(f"   ✓ models module: {models}")
    print(f"   ✓ models.__file__: {models.__file__}")
    print(f"   ✓ dir(models): {dir(models)}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n2. Testing individual model imports...")
try:
    from models import RAFT, TransformerLongContext, PatchTST
    print(f"   ✓ RAFT: {RAFT}")
    print(f"   ✓ RAFT type: {type(RAFT)}")
    print(f"   ✓ TransformerLongContext: {TransformerLongContext}")
    print(f"   ✓ TransformerLongContext type: {type(TransformerLongContext)}")
    print(f"   ✓ PatchTST: {PatchTST}")
    print(f"   ✓ PatchTST type: {type(PatchTST)}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing PatchTST module directly...")
try:
    import models.PatchTST as PatchTST_module
    print(f"   ✓ PatchTST module: {PatchTST_module}")
    print(f"   ✓ Has Model? {hasattr(PatchTST_module, 'Model')}")
    if hasattr(PatchTST_module, 'Model'):
        print(f"   ✓ PatchTST_module.Model: {PatchTST_module.Model}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n4. Testing if PatchTST is callable...")
try:
    print(f"   ✓ callable(RAFT): {callable(RAFT)}")
    print(f"   ✓ callable(TransformerLongContext): {callable(TransformerLongContext)}")
    print(f"   ✓ callable(PatchTST): {callable(PatchTST)}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n5. Testing Exp_Basic...")
try:
    from exp.exp_basic import Exp_Basic
    print(f"   ✓ Exp_Basic imported")
    
    # Check what's in the source
    import inspect
    init_source = inspect.getsource(Exp_Basic.__init__)
    print(f"   ✓ Exp_Basic.__init__ source (first 500 chars):")
    print(init_source[:500])
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
