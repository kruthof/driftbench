"""Base runner: litellm wrapper with retry, rate-limiting, and metadata capture."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import jsonlines
import litellm
import yaml
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from drift_bench.prompts.loader import render_system_prompt

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: float = 0.0
    token_count: int | None = None
    latency_ms: float | None = None
    is_probe: bool = False


@dataclass
class RunMetadata:
    """Full metadata for a single experimental run."""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    brief_id: str = ""
    model_id: str = ""
    condition: str = ""
    repetition: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    error: str | None = None


@dataclass
class Transcript:
    """Full transcript of a run, ready for serialization."""
    metadata: RunMetadata
    messages: list[Message] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "messages": [asdict(m) for m in self.messages],
        }

    def save(self, output_dir: Path) -> Path:
        """Save transcript as a JSONL append (one line per run, atomic write)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        # Use run_id in filename to avoid concurrent writes to the same file
        model_safe = self.metadata.model_id.replace("/", "_")
        fname = f"{self.metadata.run_id}_{self.metadata.brief_id}_{self.metadata.condition}_{model_safe}.jsonl"
        path = output_dir / fname
        with jsonlines.open(path, mode="w") as writer:
            writer.write(self.to_dict())
        return path


class RateLimiter:
    """Simple token-bucket rate limiter per provider."""

    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 150000):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._request_times: list[float] = []
        self._token_counts: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 1000) -> None:
        """Wait until rate limit allows another request."""
        while True:
            wait_needed = 0.0
            async with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0

                # Prune old entries
                self._request_times = [t for t in self._request_times if t > cutoff]
                self._token_counts = [(t, n) for t, n in self._token_counts if t > cutoff]

                # Check request limit
                if len(self._request_times) >= self.rpm:
                    wait_needed = max(wait_needed, self._request_times[0] - cutoff)

                # Check token limit
                current_tokens = sum(n for _, n in self._token_counts)
                if current_tokens + estimated_tokens > self.tpm and self._token_counts:
                    wait_needed = max(wait_needed, self._token_counts[0][0] - cutoff)

                if wait_needed <= 0:
                    # No wait needed — record and return
                    self._request_times.append(time.monotonic())
                    self._token_counts.append((time.monotonic(), estimated_tokens))
                    return

            # Sleep OUTSIDE the lock, then re-check
            await asyncio.sleep(wait_needed)


class BaseRunner:
    """Base class for all experimental runners."""

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.system_prompt = render_system_prompt()
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._init_rate_limiters()

        # Build model_id -> provider lookup from config
        self._provider_map: dict[str, str] = {}
        for m in self.config["models"].get("subjects", []):
            self._provider_map[m["id"]] = m.get("provider", m["id"].split("/")[0])

    def _init_rate_limiters(self) -> None:
        """Create a rate limiter per provider from config."""
        for provider, limits in self.config.get("rate_limits", {}).items():
            self._rate_limiters[provider] = RateLimiter(
                requests_per_minute=limits.get("requests_per_minute", 60),
                tokens_per_minute=limits.get("tokens_per_minute", 150000),
            )

    def _get_provider(self, model_id: str) -> str:
        """Look up provider from config, falling back to model_id prefix."""
        return self._provider_map.get(model_id, model_id.split("/")[0])

    def _get_temperature(self, model_id: str) -> float:
        """Look up temperature for a model from config."""
        for m in self.config["models"]["subjects"]:
            if m["id"] == model_id:
                return m.get("temperature", 0.7)
        return 0.7

    def _get_max_tokens(self, model_id: str) -> int:
        """Look up max_tokens for a model from config."""
        for m in self.config["models"]["subjects"]:
            if m["id"] == model_id:
                return m.get("max_tokens", 4096)
        return 4096

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((
            litellm.exceptions.RateLimitError,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.InternalServerError,
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.Timeout,
        )),
        reraise=True,
    )
    async def _call_llm(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        """Make a single LLM API call via litellm with retry and rate limiting.

        Returns dict with keys: content, input_tokens, output_tokens, cost, latency_ms
        """
        provider = self._get_provider(model_id)
        limiter = self._rate_limiters.get(provider)
        if limiter:
            await limiter.acquire()

        start = time.monotonic()
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await litellm.acompletion(**kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000

        usage = response.usage
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Cost lookup failed for {model_id}: {e}")
            cost = 0.0

        content = response.choices[0].message.content or ""

        return {
            "content": content,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cost": cost,
            "latency_ms": elapsed_ms,
        }

    def _build_messages_list(self, conversation: list[Message]) -> list[dict[str, str]]:
        """Convert internal Message list to litellm format, excluding probe messages."""
        return [
            {"role": m.role, "content": m.content}
            for m in conversation
            if not m.is_probe
        ]

    async def run(
        self,
        brief: dict[str, Any],
        model_id: str,
        condition: str,
        repetition: int = 0,
        output_dir: Path | None = None,
    ) -> Transcript:
        """Execute a full run. Subclasses must implement this."""
        raise NotImplementedError
