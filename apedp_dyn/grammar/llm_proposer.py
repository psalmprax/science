"""Optional LLM-guided proposer (seam-compliant).

Proposes candidate equation structures via external LLM APIs while respecting
proposer/referee separation: the LLM suggests hypotheses, never decides
acceptance. All candidates go through the normal gates/fitting/adjudication
pipeline.

Supports: OpenRouter, OpenAI, DeepSeek, NVIDIA NIM, HuggingFace, Aihubmix,
and custom endpoints configured by the user.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from apedp_dyn.grammar.candidate_ir import CandidateDynamics, SystemCandidate, Term
from apedp_dyn.grammar.proposer import CandidateProposer

PROPOSER_VERSION = "1.1.0-llm"


# Provider defaults
_PROVIDER_DEFAULTS = {
    "openrouter": {
        "base": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": ["OPENROUTER_API_KEY", "APEDP_LLM_KEY"],
        "model_env": "APEDP_LLM_MODEL",
        "default_model": "deepseek-chat",
    },
    "openai": {
        "base": "https://api.openai.com/v1/chat/completions",
        "key_env": ["OPENAI_API_KEY", "APEDP_LLM_KEY"],
        "model_env": "APEDP_LLM_MODEL",
        "default_model": "gpt-4o-mini",
    },
    "deepseek": {
        "base": "https://api.deepseek.com/v1/chat/completions",
        "key_env": ["DEEPSEEK_API_KEY", "APEDP_LLM_KEY"],
        "model_env": "APEDP_LLM_MODEL",
        "default_model": "deepseek-chat",
    },
    "nvidia": {
        "base": "https://integrate.api.nvidia.com/v1/chat/completions",
        "key_env": ["NVIDIA_API_KEY", "NIM_API_KEY", "APEDP_LLM_KEY"],
        "model_env": "APEDP_LLM_MODEL",
        "default_model": "deepseek-ai/deepseek-chat",
    },
    "huggingface": {
        "base": "https://api-inference.huggingface.co/models/{model}/v1/chat/completions",
        "key_env": ["HUGGINGFACE_API_KEY", "HF_TOKEN", "APEDP_LLM_KEY"],
        "model_env": "APEDP_LLM_MODEL",
        "default_model": "deepseek-ai/DeepSeek-V3",
    },
    "aihubmix": {
        "base": "https://aihubmix.com/v1/chat/completions",
        "key_env": ["AIHUBMIX_API_KEY", "APEDP_LLM_KEY"],
        "model_env": "APEDP_LLM_MODEL",
        "default_model": "deepseek-chat",
    },
    "custom": {
        "base": None,
        "key_env": ["APEDP_LLM_KEY", "CUSTOM_API_KEY"],
        "model_env": "APEDP_LLM_MODEL",
        "default_model": "gpt-3.5-turbo",
    },
}


class LLMProposer:
    """Optional LLM-guided proposer. Disabled by default."""

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        provider: str = "openrouter",  # openrouter, openai, deepseek, nvidia, huggingface, aihubmix, custom
        temperature: float = 0.7,
        max_candidates: int = 10,
        proposer_id: str = "llm-guided-v1",
        enabled: bool = False,
        headers: dict[str, str] | None = None,
    ):
        provider = provider.lower()
        self.provider = provider
        defaults = _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["custom"])
        
        # Get model
        self.model = model or os.environ.get(defaults["model_env"]) or os.environ.get("APEDP_LLM_MODEL") or defaults["default_model"]
        
        # Get API base
        if api_base:
            self.api_base = api_base
        elif os.environ.get("APEDP_LLM_BASE"):
            self.api_base = os.environ.get("APEDP_LLM_BASE")
        elif defaults["base"]:
            if provider == "huggingface" and "{model}" in defaults["base"]:
                self.api_base = defaults["base"].format(model=self.model)
            else:
                self.api_base = defaults["base"]
        else:
            self.api_base = None
        
        # Get API key
        self.api_key = api_key
        if not self.api_key:
            for key_env in defaults["key_env"]:
                self.api_key = os.environ.get(key_env)
                if self.api_key:
                    break
        
        self.temperature = temperature
        self.max_candidates = max_candidates
        self._proposer_id = proposer_id
        self.enabled = enabled
        self.headers = headers or {}

    @property
    def proposer_id(self) -> str:
        return self._proposer_id

    def propose(self, problem) -> Sequence[CandidateDynamics | SystemCandidate]:
        if not self.enabled or not self.api_key:
            return []

        try:
            return self._propose_via_api(problem)
        except Exception:
            # Fail gracefully - never break the pipeline
            return []

    def _propose_via_api(self, problem) -> Sequence[CandidateDynamics]:
        """Call LLM API to suggest equation structures."""
        try:
            import httpx
        except ImportError:
            return []

        if not self.api_key or not self.api_base:
            return []

        # Build context about the problem
        context = self._build_context(problem)
        prompt = self._build_prompt(context)

        # Build headers based on provider
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/APEDP"
            headers["X-Title"] = "APEDP"
        elif self.provider == "huggingface":
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Merge custom headers
        if self.headers:
            headers.update(self.headers)

        try:
            resp = httpx.post(
                self.api_base,
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert in dynamical systems and equation discovery. Suggest plausible equation structures."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 2000,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except Exception:
            return []

        return self._parse_suggestions(content)

    def _build_context(self, problem) -> dict[str, Any]:
        states = [s.name for s in getattr(problem, "states", [])]
        inputs = [u.name for u in getattr(problem, "inputs", [])]
        return {
            "states": states,
            "inputs": inputs,
            "name": getattr(problem, "name", "unknown"),
            "problem_id": getattr(problem, "problem_id", "unknown"),
        }

    def _build_prompt(self, context: dict[str, Any]) -> str:
        return f"""Given a dynamical system discovery problem:
- States: {context['states']}
- Inputs: {context['inputs']}
- Problem: {context['name']}

Suggest up to {self.max_candidates} plausible equation structures for xddot = f(x,v,u) in the form of term combinations.
Available term types: x, x2, x3, v, abs_v_v, v2, u, x_v, sin_x, cos_x, tanh_x, abs_x, x_u, sat_x, langmuir_bind (with scale).

For each suggestion, provide:
1. terms: list of term names
2. rationale: brief explanation
3. confidence: 0-1

Return as JSON: [{{"terms": ["x", "v"], "rationale": "...", "confidence": 0.8}}, ...]"""

    def _parse_suggestions(self, content: str) -> Sequence[CandidateDynamics]:
        candidates = []
        try:
            # Try to extract JSON from content
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                suggestions = json.loads(json_str)
                for i, sugg in enumerate(suggestions[: self.max_candidates]):
                    terms_list = sugg.get("terms", [])
                    if not terms_list:
                        continue
                    # Validate terms exist in grammar
                    from apedp_dyn.grammar.mechanical_grammar import _TERM_BUILDERS
                    valid_terms = [t for t in terms_list if t in _TERM_BUILDERS]
                    if not valid_terms:
                        continue
                    terms = tuple(Term(name, f"a{j}") for j, name in enumerate(valid_terms))
                    cand = CandidateDynamics(
                        lhs="xddot",
                        terms=terms,
                        metadata={
                            "family": "mechanical_2nd_order",
                            "proposer": self._proposer_id,
                            "llm_rationale": sugg.get("rationale", ""),
                            "llm_confidence": sugg.get("confidence", 0.5),
                        },
                        candidate_id=f"{self._proposer_id}-{i:04d}",
                        parameter_bounds={f"a{j}": (-1e6, 1e6) for j in range(len(valid_terms))},
                    )
                    candidates.append(cand)
        except Exception:
            pass
        return candidates


def create_llm_proposer(**kwargs) -> LLMProposer:
    """Factory function to create LLM proposer (disabled by default)."""
    return LLMProposer(enabled=False, **kwargs)
