import sys
import os

# Add the worktree root to sys.path BEFORE tests/ so the real finetune_tw package takes precedence
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Force reload finetune_tw if it was imported from wrong location
if 'finetune_tw' in sys.modules:
    finetune_tw = sys.modules['finetune_tw']
    if not finetune_tw.__file__.startswith(_root + '/finetune_tw'):
        # Wrong package loaded, remove it
        mods_to_remove = [k for k in sys.modules if k == 'finetune_tw' or k.startswith('finetune_tw.')]
        for mod in mods_to_remove:
            del sys.modules[mod]
