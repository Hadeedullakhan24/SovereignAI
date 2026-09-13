"""Prompt Template Loader for SovereignAI Agent Orchestration (prompt_loader.py).

Provides lightweight, dependency-free template loading and rendering for reusable
prompt templates stored in the agent/prompts/ directory.

Usage
-----
    from agent.prompts.prompt_loader import PromptLoader, render_prompt

    # Via class instance:
    loader = PromptLoader()
    rendered = loader.render("approval_note_generation", **findings)

    # Or via module convenience function:
    rendered = render_prompt("summarization", **params)
"""

from __future__ import annotations

import logging
from pathlib import Path
import string
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("sovereign_ai.prompt_loader")


class _SafeFormatter(string.Formatter):
    """String formatter that gracefully handles missing keys and None values."""

    def get_value(self, key: Union[int, str], args: Any, kwargs: Dict[str, Any]) -> Any:
        if isinstance(key, str):
            if key in kwargs:
                val = kwargs[key]
                return "" if val is None else str(val)
            # Safe fallback for missing placeholders
            return f"[{key.upper()} NOT PROVIDED]"
        return super().get_value(key, args, kwargs)


class PromptLoader:
    """Loads and renders prompt templates from the agent/prompts/ directory."""

    def __init__(self, prompts_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize PromptLoader with target prompts directory.

        Defaults to the directory containing this prompt_loader.py file.
        """
        if prompts_dir is not None:
            self.prompts_dir = Path(prompts_dir).resolve()
        else:
            self.prompts_dir = Path(__file__).resolve().parent

        self._cache: Dict[str, str] = {}
        self._formatter = _SafeFormatter()

    def _resolve_template_path(self, template_name: str) -> Path:
        """Resolve a template name to a file path."""
        clean_name = template_name.strip()
        if not clean_name.endswith(".txt"):
            clean_name = f"{clean_name}.txt"
        return self.prompts_dir / clean_name

    def load(self, template_name: str) -> str:
        """Load the raw content of a template file by name (cached in memory).

        Args:
            template_name: Filename with or without .txt extension
                           (e.g., "approval_note_generation" or "approval_note_generation.txt").

        Returns:
            Raw template string.

        Raises:
            FileNotFoundError: If the template file does not exist in prompts_dir.
        """
        path = self._resolve_template_path(template_name)
        key = str(path)

        if key in self._cache:
            return self._cache[key]

        if not path.is_file():
            raise FileNotFoundError(
                f"Prompt template '{template_name}' not found at {path}. "
                f"Available templates: {self.list_templates()}"
            )

        content = path.read_text(encoding="utf-8")
        self._cache[key] = content
        return content

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Load and render a template by replacing {placeholder} variables.

        Args:
            template_name: Name of the template to load.
            **kwargs: Dynamic values for template placeholders.

        Returns:
            Rendered string with all variables substituted.
        """
        template_str = self.load(template_name)
        try:
            return self._formatter.vformat(template_str, (), kwargs)
        except Exception as exc:
            logger.warning(
                "SafeFormatter encountered error during render of '%s': %s. Falling back to str.format()",
                template_name, exc
            )
            # Fallback
            return template_str.format(**kwargs)

    def list_templates(self) -> List[str]:
        """List all available template names in the prompts directory."""
        if not self.prompts_dir.is_dir():
            return []
        return sorted([f.stem for f in self.prompts_dir.glob("*.txt")])

    def reload(self) -> None:
        """Clear cached templates from memory."""
        self._cache.clear()


# Default singleton instance
_default_loader = PromptLoader()


def load_prompt(template_name: str) -> str:
    """Convenience function to load a template using the default PromptLoader."""
    return _default_loader.load(template_name)


def render_prompt(template_name: str, **kwargs: Any) -> str:
    """Convenience function to render a template using the default PromptLoader."""
    return _default_loader.render(template_name, **kwargs)


def get_prompt_loader(prompts_dir: Optional[Union[str, Path]] = None) -> PromptLoader:
    """Factory function to acquire a PromptLoader instance."""
    if prompts_dir is None:
        return _default_loader
    return PromptLoader(prompts_dir=prompts_dir)


if __name__ == "__main__":
    loader = PromptLoader()
    print("Available templates:", loader.list_templates())
    for t in loader.list_templates():
        content = loader.load(t)
        print(f"  [+] {t}.txt ({len(content)} chars)")
