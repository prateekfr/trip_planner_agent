from __future__ import annotations
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPModuleClient:
    """
    Generic MCP client that spawns a server by Python module path.
    Example:
        flights = MCPModuleClient(module="providers.flight_server")
        flights.call("search_flights", departure_id="BOM", arrival_id="DEL", outbound_date="2025-10-01")
    """
    def __init__(
        self,
        module: str,
        python_exe: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        args: Optional[List[str]] = None,
    ):
        self.command = python_exe or sys.executable
        self.args = args or ["-m", module]
        self.env = env  

    async def _acall(self, tool: str, arguments: Dict[str, Any]) -> Any:
        params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments=arguments)
                parts = getattr(result, "content", []) or []
                texts: List[str] = []
                for part in parts:
                    t = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
                    if t: texts.append(t)
                raw = "\n".join(texts).strip()
                try:
                    return json.loads(raw) if raw else {}
                except Exception:
                    return raw

    def call(self, tool: str, **kwargs) -> Any:
        return asyncio.run(self._acall(tool, kwargs))
    
class MCPClient(MCPModuleClient):
    """Alias for backward compatibility."""
    pass

def flights_client(env: Optional[Dict[str,str]] = None) -> MCPModuleClient:
    return MCPModuleClient("providers.flight_server", env=env)

def buses_client(env: Optional[Dict[str,str]] = None) -> MCPModuleClient:
    return MCPModuleClient("providers.bus_server", env=env)

def trains_client(env: Optional[Dict[str,str]] = None) -> MCPModuleClient:
    return MCPModuleClient("providers.train_server", env=env)

def hotels_client(env: Optional[Dict[str,str]] = None) -> MCPModuleClient:
    return MCPModuleClient("providers.hotel_server", env=env)
