from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import BaseSchema
from .core import McpTool


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
