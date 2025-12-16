from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..schemas.domain import CapabilityName
from .base import Capability, CapabilityContext, CapabilityResult


@dataclass(frozen=True)
class SummarizeCapability(Capability):
    name: CapabilityName = CapabilityName.summarize

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        text = str(args.get("text") or "")
        redacted = ctx.policy.redact(text)
        if ctx.policy.detect_prompt_injection(redacted):
            return CapabilityResult(ok=False, output={"error": "prompt injection detected"})
        summary = redacted[:200]
        return CapabilityResult(ok=True, output={"summary": summary})


@dataclass(frozen=True)
class WebSearchCapability(Capability):
    name: CapabilityName = CapabilityName.web_search

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        query = str(args.get("query") or args.get("q") or "").strip()
        if not query:
            return CapabilityResult(ok=False, output={"error": "missing query"})

        fn = getattr(ctx.deps, "web_search", None)
        if fn is None:
            return CapabilityResult(ok=False, output={"error": "web_search not configured"})

        result = await fn(**args)
        if isinstance(result, dict):
            return CapabilityResult(ok=True, output=result)
        return CapabilityResult(ok=True, output={"result": result})


@dataclass(frozen=True)
class ShellExecCapability(Capability):
    name: CapabilityName = CapabilityName.shell_exec

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        command = str(args.get("command") or args.get("cmd") or "").strip()
        if not command:
            return CapabilityResult(ok=False, output={"error": "missing command"})

        fn = getattr(ctx.deps, "shell_exec", None)
        if fn is None:
            return CapabilityResult(ok=False, output={"error": "shell_exec not configured"})

        result = await fn(**args)
        if isinstance(result, dict):
            return CapabilityResult(ok=True, output=result)
        return CapabilityResult(ok=True, output={"result": result})


@dataclass(frozen=True)
class WebFetchCapability(Capability):
    name: CapabilityName = CapabilityName.web_fetch

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        url = str(args.get("url") or "").strip()
        if not url:
            return CapabilityResult(ok=False, output={"error": "missing url"})

        fn = getattr(ctx.deps, "web_fetch", None)
        if fn is None:
            return CapabilityResult(ok=False, output={"error": "web_fetch not configured"})

        result = await fn(**args)
        if isinstance(result, dict):
            return CapabilityResult(ok=True, output=result)
        return CapabilityResult(ok=True, output={"content": result})


@dataclass(frozen=True)
class CodeExecutionCapability(Capability):
    name: CapabilityName = CapabilityName.code_execution

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        language = str(args.get("language") or "").strip().lower()
        code = str(args.get("code") or "")
        if not code:
            return CapabilityResult(ok=False, output={"error": "missing code"})

        if language in {"bash", "sh", "shell"}:
            fn = getattr(ctx.deps, "shell_exec", None)
            if fn is None:
                return CapabilityResult(ok=False, output={"error": "shell_exec not configured"})

            result = await fn(**args)
            if isinstance(result, dict):
                return CapabilityResult(ok=True, output=result)
            return CapabilityResult(ok=True, output={"result": result})

        fn = getattr(ctx.deps, "code_execute", None)
        if fn is None:
            return CapabilityResult(ok=False, output={"error": "code_execute not configured"})

        result = await fn(**args)
        if isinstance(result, dict):
            return CapabilityResult(ok=True, output=result)
        return CapabilityResult(ok=True, output={"result": result})


@dataclass(frozen=True)
class CodegenCapability(Capability):
    name: CapabilityName = CapabilityName.codegen

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        prompt = str(args.get("prompt") or args.get("instruction") or "").strip()
        if not prompt:
            return CapabilityResult(ok=False, output={"error": "missing prompt"})

        fn = getattr(ctx.deps, "codegen", None)
        if fn is None:
            return CapabilityResult(ok=False, output={"error": "codegen not configured"})

        result = await fn(**args)
        if isinstance(result, dict):
            return CapabilityResult(ok=True, output=result)
        return CapabilityResult(ok=True, output={"code": result})
