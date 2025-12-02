from __future__ import annotations

from typing import Any, Dict, Optional, List

from pydantic import Field

from .base import BaseSchema
from .core import McpTool


class ToolMetadata(BaseSchema):
    """
    Minimal tool metadata used by strategies and tests.

    This aligns loosely with MCP tool descriptions and gateway listings.
    """

    name: str = Field(..., description="Tool name", min_length=1, max_length=128)
    description: Optional[str] = Field(
        None, description="Short human-readable description of the tool.", max_length=2048
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameters schema for the tool. When sourced from MCP servers, this" " may be called 'inputSchema'."
        ),
    )


class ToolResult(BaseSchema):
    """
    Result wrapper for tool invocation.
    """

    ok: bool = Field(True, description="Whether the tool invocation succeeded.")
    data: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary JSON payload returned by the tool.")


class ToolsPage(BaseSchema):
    """
    Paginated tools result used by strategies/clients when backends support pagination.

    - items: list of tools for this page
    - next_cursor: opaque cursor to request the next page; None means no further pages

    Example (sync):
        page = client.list_tools_page("server-1", limit=50)
        tools = list(page.items)
        while page.next_cursor:
            page = client.list_tools_page("server-1", cursor=page.next_cursor, limit=50)
            tools.extend(page.items)

    Example (async):
        page = await aclient.list_tools_page("server-1", limit=50)
        tools = list(page.items)
        while page.next_cursor:
            page = await aclient.list_tools_page("server-1", cursor=page.next_cursor, limit=50)
            tools.extend(page.items)
    """

    items: List[McpTool] = Field(default_factory=list, description="Tools returned for the current page.")
    next_cursor: Optional[str] = Field(
        default=None, alias="nextCursor", description="Opaque cursor for fetching the next page, if available."
    )
