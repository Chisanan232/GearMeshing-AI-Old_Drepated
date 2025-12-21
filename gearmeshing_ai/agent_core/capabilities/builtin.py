from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..schemas.domain import CapabilityName
from .base import Capability, CapabilityContext, CapabilityResult


@dataclass(frozen=True)
class SummarizeCapability(Capability):
    """
    Capability to summarize text content.

    This capability provides text summarization functionality. It incorporates
    safety mechanisms to redact sensitive information and detect prompt injection
    attacks before processing the text.
    """

    name: CapabilityName = CapabilityName.summarize

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        """
        Execute the summarization logic.

        Args:
            ctx: The execution context containing dependencies and policy.
            args: Dictionary of arguments:
                - text (str): The input text to summarize.

        Returns:
            CapabilityResult:
                - Success: output contains {"summary": "..."}
                - Failure: output contains {"error": "..."} (e.g., injection detected)
        """
        text = str(args.get("text") or "")
        redacted = ctx.policy.redact(text)
        if ctx.policy.detect_prompt_injection(redacted):
            return CapabilityResult(ok=False, output={"error": "prompt injection detected"})
        summary = redacted[:200]
        return CapabilityResult(ok=True, output={"summary": summary})


@dataclass(frozen=True)
class WebSearchCapability(Capability):
    """
    Capability to perform web searches.

    Delegates the search operation to the configured ``web_search`` dependency function.
    """

    name: CapabilityName = CapabilityName.web_search

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        """
        Execute a web search.

        Args:
            ctx: The execution context.
            args: Dictionary of arguments:
                - query (str): The search query. Alias: 'q'.

        Returns:
            CapabilityResult: The search results from the provider, or an error if not configured.
        """
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
    """
    Capability to execute shell commands.

    Allows the agent to run system commands via the ``shell_exec`` dependency.
    This is a high-risk capability and is typically guarded by strict policies.
    """

    name: CapabilityName = CapabilityName.shell_exec

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        """
        Execute a shell command.

        Args:
            ctx: The execution context.
            args: Dictionary of arguments:
                - command (str): The command string to execute. Alias: 'cmd'.

        Returns:
            CapabilityResult: Standard output/error of the command execution.
        """
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
    """
    Capability to fetch web page content.

    Retrieves the content of a specific URL using the ``web_fetch`` dependency.
    """

    name: CapabilityName = CapabilityName.web_fetch

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        """
        Fetch content from a URL.

        Args:
            ctx: The execution context.
            args: Dictionary of arguments:
                - url (str): The target URL to fetch.

        Returns:
            CapabilityResult: The page content or an error message.
        """
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
class DocsReadCapability(Capability):
    """
    Capability to read documentation.

    Fetches documentation from internal or external sources via the ``docs_read`` dependency.
    """

    name: CapabilityName = CapabilityName.docs_read

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        """
        Read documentation.

        Args:
            ctx: The execution context.
            args: Implementation-specific arguments passed to the docs provider.

        Returns:
            CapabilityResult: The requested documentation content.
        """
        fn = getattr(ctx.deps, "docs_read", None)
        if fn is None:
            return CapabilityResult(ok=False, output={"error": "docs_read not configured"})

        result = await fn(**args)
        if isinstance(result, dict):
            return CapabilityResult(ok=True, output=result)
        return CapabilityResult(ok=True, output={"result": result})


@dataclass(frozen=True)
class CodeExecutionCapability(Capability):
    """
    Capability to execute arbitrary code.

    Supports running code in various languages (e.g., Python, Bash).
    Delegates to ``code_execute`` or ``shell_exec`` depending on the language.
    """

    name: CapabilityName = CapabilityName.code_execution

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        """
        Execute code snippet.

        Args:
            ctx: The execution context.
            args: Dictionary of arguments:
                - language (str): The programming language (e.g., 'python', 'bash').
                - code (str): The source code to execute.

        Returns:
            CapabilityResult: The execution result (stdout, stderr, return code).
        """
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
    """
    Capability to generate code using an AI model.

    Uses the ``codegen`` dependency to produce code based on a prompt or instruction.
    """

    name: CapabilityName = CapabilityName.codegen

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        """
        Generate code.

        Args:
            ctx: The execution context.
            args: Dictionary of arguments:
                - prompt (str): Description of the code to generate. Alias: 'instruction'.

        Returns:
            CapabilityResult: The generated code content.
        """
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


@dataclass(frozen=True)
class McpCallCapability(Capability):
    """
    Capability to call tools via the Model Context Protocol (MCP).

    This capability acts as a bridge to the MCP ecosystem, allowing the agent
    to invoke tools exposed by connected MCP servers.
    """

    name: CapabilityName = CapabilityName.mcp_call

    async def execute(self, ctx: CapabilityContext, *, args: Dict[str, Any]) -> CapabilityResult:
        """
        Call an MCP tool.

        Args:
            ctx: The execution context.
            args: Dictionary of arguments:
                - server_id (str): The ID of the MCP server.
                - tool_name (str): The name of the tool to invoke.
                - tool_args (dict): Arguments for the tool. Alias: 'args'.

        Returns:
            CapabilityResult: The tool execution result from the MCP server.
        """
        mcp_call = ctx.deps.mcp_call
        if mcp_call is None:
            return CapabilityResult(ok=False, output={"error": "mcp_call not configured"})

        server_id = str(args.get("server_id") or "").strip()
        tool_name = str(args.get("tool_name") or "").strip()
        tool_args = args.get("tool_args")
        if tool_args is None:
            tool_args = args.get("args")
        if tool_args is None:
            tool_args = {}

        if not server_id:
            return CapabilityResult(ok=False, output={"error": "missing server_id"})
        if not tool_name:
            return CapabilityResult(ok=False, output={"error": "missing tool_name"})
        if not isinstance(tool_args, dict):
            return CapabilityResult(ok=False, output={"error": "tool_args must be a dict"})

        res = await mcp_call(server_id, tool_name, dict(tool_args))
        ok = bool(getattr(res, "ok", True))
        data = getattr(res, "data", None)
        if isinstance(data, dict):
            return CapabilityResult(ok=ok, output=data)
        if hasattr(res, "model_dump"):
            return CapabilityResult(ok=ok, output=res.model_dump())
        return CapabilityResult(ok=ok, output={"result": res})
