from __future__ import annotations
from typing import Any, Dict, Iterable, Optional

class Scan(dict):
    """
    Dict-of-dicts container:
      scan[scan_number][column_name] -> array or value

    Example keys per scan:
      images, energy, bragg, braggOffset, correction, orig1,
      averaged (bool), normalised (bool), path (str), detector (str)
    """
    def add_scan(self, scan_number: Any, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        self[scan_number] = data

    def next_index(self) -> int:
        """Returns the next integer scan index (0-based)."""
        ints = [k for k in self.keys() if isinstance(k, int)]
        return (max(ints) + 1) if ints else 0

    def latest(self) -> Optional[Any]:
        return next(reversed(self)) if self else None

    def columns(self, scan_number: Any) -> Iterable[str]:
        return self[scan_number].keys()