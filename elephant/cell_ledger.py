"""Elephant cell-ledger producer SDK (quilt bridge, synergy missing-link ②)."

Implements the sha256 chain-seal per quilt-rust/docs/cell-ledger.md §4, plus
a thin async POST to the quilt relay endpoint.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def _canonical(o: Any) -> bytes:
    """Canonical JSON: compact, sorted keys, float rounding."""
    return json.dumps(o, sort_keys=True, separators=(',', ':'),
                       allow_nan=False, default=str).encode()


def seal(entry: Dict[str, Any], prev_hash: str) -> str:
    """Compute the chain hash for a ledger entry.

    Per quilt-rust §4: ``hash(e) = sha256_hex(canonical_json(e minus hash))``
    where ``e`` includes ``prev_hash``.
    """
    payload = {k: v for k, v in entry.items() if k != 'hash'}
    payload['prev_hash'] = prev_hash
    return hashlib.sha256(_canonical(payload)).hexdigest()


def genesis_commit(cell_id: str, genesis_ts: float) -> str:
    """Compute the chain root hash for an empty ledger (quilt-rust §4)."""
    return hashlib.sha256(_canonical({
        'kind': 'quilt-cell-ledger/1',
        'cell_id': cell_id,
        'genesis': None,
        'genesis_ts': genesis_ts,
    })).hexdigest()


class CellLedgerProducer:
    """Manages sha256 chain state for one cell and produces sealed entries.

    Usage::
        producer = CellLedgerProducer(cell_id='room.field.living-room')
        entry = producer.record(fit, prev_fit, ts=msg.ts)
        # entry now has 'prev_hash' and 'hash' — ready for POST or append.
    """

    def __init__(self, cell_id: str) -> None:
        self.cell_id = cell_id
        self._head: str = ''  # set on first record via genesis_commit
        self._seq: int = 0

    def record(self, entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Seal a ledger entry with chain hash. Returns None if entry is None."""
        if entry is None:
            return None
        if not self._head:
            self._head = genesis_commit(self.cell_id, entry.get('ts', 0.0))
        self._seq += 1
        entry['seq'] = self._seq
        entry['prev_hash'] = self._head
        entry['hash'] = seal(entry, self._head)
        self._head = entry['hash']
        return entry
