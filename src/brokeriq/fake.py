"""Deterministic stand-in for the LLM layer.

Used by the CLI in --offline mode and by integration tests so the full
pipeline can be exercised with zero API keys. Each stage returns the same
shape the real model would.
"""

import json


class FakeLLM:
    """Returns canned, stage-appropriate JSON for the BrokerIQ pipeline."""

    def __init__(self, *, verdict: str = "qualified", icp_score: float = 82.0) -> None:
        self.verdict = verdict
        self.icp_score = icp_score

    async def complete(self, messages, model=None, temperature=0.0, json_mode=False, max_tokens=None):
        content = self._respond(messages)
        if json_mode:
            return json.dumps(content)
        return str(content)

    async def complete_json(self, messages, model=None, temperature=0.0):
        return self._respond(messages)

    def _respond(self, messages) -> dict:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")

        if "routing supervisor" in system or "routing" in system:
            return {"next": self._supervisor_next(user)}

        if "research agent" in system:
            return {
                "summary": f"{self._company(user)} is a US-based {self._industry(user)} company "
                "with a growing commercial footprint.",
                "sources": ["https://example.com/company"],
                "naics_code": "5415",
                "naics_label": "Computer Systems Design",
                "headcount_estimate": "50-100",
                "funding_signals": [],
                "risk_flags": [],
            }

        if "qualification agent" in system:
            return {
                "icp_score": self.icp_score,
                "carrier_fit": {
                    "lines": ["workers_comp", "cyber_liability", "general_liability"],
                    "blockers": [],
                    "confidence": 0.9,
                },
                "risk_flags": [],
                "verdict": self.verdict,
            }

        if "report writer" in system:
            return {
                "headline": f"Qualified lead: {self._company(user)}",
                "summary": "Strong ICP fit; three coverage lines quoted.",
                "outreach_angle": "Lead with cyber liability renewal timeline.",
                "recommended_action": "Call within 48 hours.",
            }

        if "memory extractor" in system:
            return {
                "ops": [
                    {
                        "op": "ADD",
                        "namespace": ["leads", self._company(user).lower().replace(" ", "-")],
                        "key": "profile",
                        "value": {"naics": "5415", "verdict": self.verdict, "score": self.icp_score},
                    }
                ]
            }

        return {"next": "done"}

    @staticmethod
    def _supervisor_next(user: str) -> str:
        """Parse the state summary lines and route like the rule fallback would."""
        lines = {}
        for line in user.splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                lines[key.strip()] = value.strip()

        if lines.get("research") != "done":
            return "research"
        if lines.get("qualification") == "missing":
            return "qualification"
        if lines.get("brief") != "done":
            return "report"
        if lines.get("memory") != "done":
            return "memory"
        return "done"

    @staticmethod
    def _company(user: str) -> str:
        for line in user.splitlines():
            if line.startswith("Company:"):
                return line.split(":", 1)[1].strip()
        return "unknown"

    @staticmethod
    def _industry(user: str) -> str:
        for line in user.splitlines():
            if line.startswith("Reported industry:"):
                return line.split(":", 1)[1].strip()
        return "technology"
