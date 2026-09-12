"""The orchestrator. Routes an input to Tier 0, 1 or 2 and assembles the output.

Two production concerns handled here that are easy to miss:

  Sticky abstain. Sellers re-upload the same image with a different filename
  until they get a number. Hash the input; if the gate already rejected it,
  return the same answer rather than rolling the dice on VLM nondeterminism.

  Reason-code logging. Every refusal is labelled data. Tier 0 false positives —
  real trucks the gate rejected — are the highest-value training examples you
  will ever collect, and the ones that quietly kill retention.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .gate import Detector, PhotoInput, run_gate
from .identity import apply_vin_decode, cross_check, decode_vin, merge_claims, \
    vin_check_digit_ok
from .ledger import build_ledger
from .pricing import QuantilePricer
from .priors import DEFAULT_PRIORS, PriorSet
from .report import (
    MAX_EXCESS_RATIO,
    condition_report,
    gate_message,
    reasoning,
    tier1_message,
)
from .schema import (
    Evidence,
    GateReason,
    SellerClaims,
    Tier,
    Valuation,
)
from .uncertainty import attribute_variance, run_uncertainty

log = logging.getLogger("truckval")

PerceptionFn = Callable[[Sequence[PhotoInput]], Evidence]


@dataclass
class Pipeline:
    detector: Detector
    perceive: PerceptionFn
    pricer: QuantilePricer
    priors: PriorSet = field(default_factory=lambda: DEFAULT_PRIORS)
    corpus_hashes: Sequence[int] = ()
    use_vpic: bool = False
    n_draws: int = 2_000
    _abstain_cache: dict[str, Valuation] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def _input_key(self, photos: Sequence[PhotoInput]) -> str:
        h = hashlib.sha256()
        for p in sorted(photos, key=lambda x: x.sha256):
            h.update(p.sha256.encode())
        return h.hexdigest()

    def run(
        self,
        photos: Sequence[PhotoInput],
        claims: Optional[SellerClaims] = None,
        plates: Optional[Sequence[Optional[str]]] = None,
    ) -> Valuation:
        claims = claims or SellerClaims()
        key = self._input_key(photos)
        if key in self._abstain_cache:
            log.info("sticky_abstain key=%s", key[:12])
            return self._abstain_cache[key]

        # ---------------------------------------------------------- gate
        gate = run_gate(photos, self.detector, plates, self.corpus_hashes)
        if gate.tier == Tier.NOTHING:
            v = Valuation(
                tier=Tier.NOTHING,
                gate=gate,
                evidence=Evidence(),
                message=gate_message(gate),
            )
            _assert_no_anchor_leak(v)
            self._abstain_cache[key] = v
            self._log(v, key)
            return v

        # ---------------------------------------------------- perception
        evidence = self.perceive(photos)
        if claims.region:
            evidence.region = claims.region

        # ------------------------------------------------------ identity
        if evidence.identity.vin.known:
            vin = str(evidence.identity.vin.value)
            if vin_check_digit_ok(vin) and self.use_vpic:
                decoded = decode_vin(vin)
                if decoded:
                    evidence = apply_vin_decode(evidence, decoded)

        contradictions = cross_check(evidence, claims)
        evidence = merge_claims(evidence, claims)

        # ------------------------------------------------- tier decision
        if not evidence.identity.resolved:
            gate.reasons.append(GateReason.NO_IDENTITY)
            return self._tier1(gate, evidence, contradictions, key)

        if gate.tier == Tier.CONDITION_ONLY:
            return self._tier1(gate, evidence, contradictions, key)

        # ------------------------------------------------------- pricing
        result = run_uncertainty(
            evidence, self.pricer, self.priors, n_draws=self.n_draws,
            contradicted=frozenset(_claim_path(c.field) for c in contradictions),
        )

        if result.excess_ratio > MAX_EXCESS_RATIO:
            gate.reasons.append(GateReason.INTERVAL_TOO_WIDE)
            return self._tier1(gate, evidence, contradictions, key)

        ledger = build_ledger(evidence)
        next_photos = attribute_variance(evidence, self.pricer, self.priors)

        v = Valuation(
            tier=Tier.PRICED,
            gate=gate,
            evidence=evidence,
            price=result.price,
            baseline=result.baseline_median,
            ledger=ledger,
            contradictions=contradictions,
            next_photos=next_photos,
            condition_report=condition_report(evidence),
            reasoning=reasoning(
                result.baseline_median, ledger, result.price, evidence,
                result.sampled_fields, contradictions,
            ),
        )
        _assert_no_anchor_leak(v)
        self._log(v, key)
        return v

    # ------------------------------------------------------------------
    def _tier1(self, gate, evidence, contradictions, key: str) -> Valuation:
        next_photos = attribute_variance(evidence, self.pricer, self.priors)
        v = Valuation(
            tier=Tier.CONDITION_ONLY,
            gate=gate,
            evidence=evidence,
            contradictions=contradictions,
            next_photos=next_photos,
            condition_report=condition_report(evidence),
            message=tier1_message(evidence, gate, next_photos),
        )
        _assert_no_anchor_leak(v)
        self._log(v, key)
        return v

    def _log(self, v: Valuation, key: str) -> None:
        log.info(json.dumps({
            "key": key[:16],
            "tier": v.tier.value,
            "reasons": [r.value for r in v.gate.reasons],
            "detected": v.gate.detected_object,
            "identified": v.evidence.identity.resolved,
            "unknowns": len(v.evidence.unknown_fields()),
            "width": v.price.width if v.price else None,
            "contradictions": [c.field for c in v.contradictions],
        }))


# --------------------------------------------------------------------------
# Outbound guardrail
# --------------------------------------------------------------------------

_CLAIM_FIELD_PATHS = {
    "year": "identity.year",
    "make": "identity.make",
    "model": "identity.model",
    "odometer_km": "condition.odometer_km",
    "box_length_ft": "configuration.box_length_ft",
}

# Anything that could be read as a number of dollars. A refusal that leaks a
# ballpark has priced the truck, and priced it worse than the model would have.
_MONEY = re.compile(r"[$\u00a3\u20ac]\s?\d|\b\d{1,3}(,\d{3})+\b|\b\d+\s?k\b", re.I)


def _claim_path(field: str) -> str:
    return _CLAIM_FIELD_PATHS.get(field, field)


def _assert_no_anchor_leak(v: Valuation) -> None:
    """Enforce the no-anchor rule in code, not in the copywriting.

    report.py is careful never to put a ballpark in a Tier 0 or Tier 1 message.
    Careful is not the same as guaranteed, and this is the one bug class where
    a single bad string undoes the entire refusal design. So it is asserted at
    the boundary instead of trusted upstream.
    """
    if v.tier != Tier.PRICED:
        if v.price is not None:
            raise AssertionError(f"tier {v.tier.value} carries a price range")
        if v.baseline is not None:
            raise AssertionError(f"tier {v.tier.value} carries a baseline")
        for name in ("message", "reasoning", "condition_report"):
            text = getattr(v, name, "") or ""
            if _MONEY.search(text):
                raise AssertionError(
                    f"tier {v.tier.value} {name} leaks a monetary anchor"
                )
    if v.tier == Tier.PRICED and v.price is None:
        raise AssertionError("tier 2 with no price range")
