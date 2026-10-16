"""Profile registry."""

from harness.profiles.app_builder import AppBuilderProfile
from harness.profiles.base import AgentConfig, BaseProfile, ProfileConfig
from harness.profiles.reasoning import ReasoningProfile
from harness.profiles.swe_bench import SWEBenchProfile
from harness.profiles.terminal import TerminalProfile

PROFILES: dict[str, type[BaseProfile]] = {
    "app-builder": AppBuilderProfile,
    "terminal": TerminalProfile,
    "swe-bench": SWEBenchProfile,
    "reasoning": ReasoningProfile,
}


def get_profile(name: str, cfg: ProfileConfig | None = None) -> BaseProfile:
    """Get a profile instance by name, optionally with custom config."""
    cls = PROFILES.get(name)
    if cls is None:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile: {name}. Available: {available}")
    return cls(cfg=cfg)


def list_profiles() -> list[dict[str, str]]:
    """List all available profiles."""
    return [
        {"name": cls().name(), "description": cls().description()}
        for cls in PROFILES.values()
    ]
