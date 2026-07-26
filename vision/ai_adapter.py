"""Optional adapter boundary for an existing learned policy.

No model dependency is imported by the OpenCV path.  Set ``AI_POLICY_FACTORY``
to ``package.module:create_policy``; the factory receives ``shared_data`` and
returns an object with ``predict(frame_bgr, context)``.  The prediction must be a
mapping with physical ``thrust``, ``roll_rate``, ``pitch_rate``, and ``yaw_rate``.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Optional


REQUIRED_ACTION_FIELDS = ("thrust", "roll_rate", "pitch_rate", "yaw_rate")


def load_policy_factory(spec: Optional[str], shared_data: dict) -> Optional[Any]:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("AI_POLICY_FACTORY must be 'package.module:factory_name'")
    module_name, attribute = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory(shared_data)


def validate_ai_action(action: Any) -> dict[str, float]:
    if not isinstance(action, Mapping):
        raise TypeError("AI policy must return a mapping of physical control values")
    missing = [field for field in REQUIRED_ACTION_FIELDS if field not in action]
    if missing:
        raise ValueError(f"AI action missing fields: {', '.join(missing)}")
    return {field: float(action[field]) for field in REQUIRED_ACTION_FIELDS}
