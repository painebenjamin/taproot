from __future__ import annotations

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Server

__all__ = [
    "get_in_memory_server", 
    "set_in_memory_server",
    "unset_in_memory_server",
    "in_memory_server_exists",
]

IN_MEMORY_SERVERS: Dict[int, Server] = {}

def get_in_memory_server(server_port: int) -> Server:
    """
    Get an in-memory server by its port number.
    """
    try:
        return IN_MEMORY_SERVERS[server_port]
    except KeyError:
        raise ValueError(f"In-memory server with port {server_port} does not exist")

def set_in_memory_server(server_port: int, server: Server) -> None:
    """
    Set an in-memory server by its port number.
    """
    if server_port in IN_MEMORY_SERVERS:
        raise ValueError(f"In-memory server with port {server_port} already exists")
    from taproot.util import logger
    logger.debug(f"Adding in-memory server with port {server_port}")
    IN_MEMORY_SERVERS[server_port] = server

def unset_in_memory_server(server_port: int) -> None:
    """
    Delete an in-memory server by its port number.
    """
    try:
        del IN_MEMORY_SERVERS[server_port]
        from taproot.util import logger
        logger.debug(f"Removed in-memory server with port {server_port}")
    except KeyError:
        raise ValueError(f"In-memory server with port {server_port} does not exist")

def in_memory_server_exists(server_port: int) -> bool:
    """
    Check if an in-memory server exists by its port number.
    """
    from taproot.util import logger
    return server_port in IN_MEMORY_SERVERS
