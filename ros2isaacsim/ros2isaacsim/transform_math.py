"""Transform helpers shared by Isaac-facing publishers."""
from __future__ import annotations


def child_relative_to_parent(parent_world, child_world):
    """Return a child transform in its parent frame using USD row-vector order."""
    return child_world * parent_world.GetInverse()
