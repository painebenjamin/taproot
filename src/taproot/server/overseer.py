from __future__ import annotations

import asyncio

from typing import Any, Optional, Dict, List, Tuple, Union, Iterator, cast

from contextlib import contextmanager
from omegaconf.basecontainer import BaseContainer

from ..payload import *
from ..client import Client, DispatcherConfigClient
from ..config import OverseerConfig
from ..util import (
    logger,
    generate_id,
    is_absolute_address,
    get_absolute_address_from_relative,
)

from .config import ConfigServer, ConfigType

__all__ = ["Overseer"]

class Overseer(ConfigServer):
    """
    Overseer is a server that manages the server instances.
    Sends payload metadata to dispatcher instances, and uses
    the score to determine which dispatcher to send the payload to.
    """
    config_class = OverseerConfig
    cached_hash: int
    dispatchers: Dict[str, Tuple[Client, bool]]
    capabilities: Dict[str, CapabilityPayload]

    _cached_hash: int
    _cached_cluster_capability: CapabilityPayload

    def __init__(self, config: ConfigType=None) -> None:
        """
        Initializes the overseer.
        """
        super().__init__(config=config)
        self.dispatchers = {}
        self.capabilities = {}

    """Public Methods"""

    def register_dispatcher(self, address: str, resolve_addresses: bool = True) -> None:
        """
        Registers a dispatcher with the overseer.
        """
        if address in self.dispatchers:
            return
        client = self.get_client_for_address(address)
        self.dispatchers[address] = (client, resolve_addresses)
        logger.info(f"Added dispatcher {address}.")
        asyncio.create_task(self.update_cluster_capability(client))

    def register_dispatcher_from_config(
        self,
        config: Union[str, Dict[str, Any], BaseContainer],
    ) -> None:
        """
        Registers a dispatcher with the overseer from a configuration.
        """
        client = DispatcherConfigClient(config)
        if client.address in self.dispatchers:
            return

        if self.use_control_encryption and self.control_encryption_key:
            client.use_control_encryption = True
            client.control_encryption_key = self.control_encryption_key

        self.dispatchers[client.address] = (client, bool(client.config.resolve_addresses))

        logger.info(f"Added dispatcher {client.address}.")
        asyncio.create_task(self.update_cluster_capability(client))

    def unregister_dispatcher(self, address: str) -> None:
        """
        Unregisters a dispatcher with the overseer.
        """
        if address not in self.dispatchers:
            raise ValueError(f"Dispatcher {address} is not registered.")
        self.dispatchers.pop(address, None)
        self.capabilities.pop(address, None)

    def unregister_all_dispatchers(self) -> None:
        """
        Unregisters all dispatchers with the overseer.
        """
        self.dispatchers.clear()
        self.capabilities.clear()

    async def update_cluster_capability(self, client: Client) -> None:
        """
        Updates the cluster capability with the dispatcher's capability.
        """
        try:
            capability = await client(self.pack_control_message("capability"), timeout=1.0, retries=0)
            self.capabilities[client.address] = capability
        except Exception as e:
            logger.warning(f"Failed to update cluster capability for {client.address}: {e}")
            # Retry in a bit
            await asyncio.sleep(15.0)
            return await self.update_cluster_capability(client)

    @property
    def dispatcher_score_timeout(self) -> Optional[float]:
        """
        Returns the timeout for requests to a dispatcher when scoring.
        """
        if hasattr(self, "_dispatcher_score_timeout"):
            return self._dispatcher_score_timeout
        configured_timeout = self.config.dispatcher_score_timeout
        if configured_timeout is not None:
            return float(configured_timeout)
        return None

    @dispatcher_score_timeout.setter
    def dispatcher_score_timeout(self, value: Optional[float]) -> None:
        """
        Sets the timeout for requests to a dispatcher when scoring.
        """
        self._dispatcher_score_timeout = value

    @property
    def dispatcher_prepare_timeout(self) -> Optional[float]:
        """
        Returns the timeout for requests to a dispatcher when preparing.
        """
        if hasattr(self, "_dispatcher_prepare_timeout"):
            return self._dispatcher_prepare_timeout
        configured_timeout = self.config.dispatcher_prepare_timeout
        if configured_timeout is not None:
            return float(configured_timeout)
        return None

    @dispatcher_prepare_timeout.setter
    def dispatcher_prepare_timeout(self, value: Optional[float]) -> None:
        """
        Sets the timeout for requests to a dispatcher when preparing.
        """
        self._dispatcher_prepare_timeout = value

    @property
    def dispatchers_hash(self) -> int:
        """
        Returns a hash of the dispatchers in the capability dictionary.
        """
        return hash(tuple(self.dispatchers.keys()))

    @property
    def cluster_capability(self) -> CapabilityPayload:
        """
        Returns the maximum capability of the cluster.
        """
        current_hash = self.dispatchers_hash
        if hasattr(self, "_cluster_capability") and hasattr(self, "_cached_hash"):
            if self._cached_hash == current_hash:
                return self._cluster_capability

        self._cluster_capability: CapabilityPayload = {
            "gpu_memory_bandwidth_gb_s": 0.0,
            "gpu_half_float_performance_gflop_s": 0.0,
            "gpu_single_float_performance_gflop_s": 0.0,
            "gpu_double_float_performance_gflop_s": 0.0,
        }
        for capability in self.capabilities.values():
            for key, value in capability.items():
                self._cluster_capability[key] = max( # type: ignore
                    self._cluster_capability[key], # type: ignore[literal-required]
                    value,
                )
        self._cached_hash = current_hash
        return self._cluster_capability

    @contextmanager
    def context(self) -> Iterator[None]:
        """
        Runtime context for the overseer.
        """
        with super().context():
            if self.config.dispatchers:
                for dispatcher in self.config.dispatchers:
                    self.register_dispatcher_from_config(dispatcher)
            yield

    """Overrides"""

    async def command(self, request: str, data: Any=None) -> Any:
        """
        Handles a control from a remote client.
        If this method is called, the client has been authenticated
        for performing administrative tasks.
        """
        if request == "register":
            assert isinstance(data, str), "Data must be a string."
            return self.register_dispatcher(data)
        elif request == "unregister":
            assert isinstance(data, str), "Data must be a string."
            return self.unregister_dispatcher(data)
        return await super().command(request, data)

    async def status(self, data: Any=None) -> OverseerStatusPayload:
        """
        Returns the status of the overseer.
        """
        status = cast(OverseerStatusPayload, await super().status(data))
        # Gather list of dispatchers
        to_query: List[Tuple[str, Tuple[Client, bool]]] = list(self.dispatchers.items())
        # Query dispatchers for status in parallel
        statuses = await asyncio.gather(
            *(
                client.command(
                    "status",
                    data=data,
                    error_response=None
                )
                for _, (client, _) in to_query
            )
        )
        # Populate status with dispatcher statuses
        status["dispatchers"] = dict(
            zip(
                (address for address, _ in to_query),
                statuses,
            )
        )
        return status

    async def handle(self, request: Any) -> Any:
        """
        Handles a request from a client.
        """
        # Validate request
        if not isinstance(request, dict):
            raise ValueError(f"Request must be a dictionary, got {type(request)}.")
        request_task = request.get("task", None)
        if request_task is None:
            raise ValueError("Request must have a task.")

        # Validate overseer state
        if not self.dispatchers:
            raise ValueError("No dispatchers are registered with the overseer.")

        # Assemble scoring request for dispatchers
        if request.get("id", None) is None:
            request["id"] = generate_id()

        request["cluster_capability"] = self.cluster_capability
        dispatcher_payload = self.pack_control_message("score", request)


        # Send scoring request to dispatchers
        dispatchers = list(self.dispatchers.values())
        dispatcher_scores = await asyncio.gather(
            *(
                dispatcher(
                    dispatcher_payload,
                    timeout=self.dispatcher_score_timeout,
                    error_response=0, # Includes timeout or other errors
                )
                for (dispatcher, resolve) in dispatchers
            )
        )

        # Choose the highest scoring dispatcher
        highest_score_index = -1
        highest_score = 0.0
        for index, score in enumerate(dispatcher_scores):
            if not isinstance(score, float):
                try:
                    score = float(score)
                except ValueError:
                    logger.warning(f"Dispatcher {dispatchers[index][0].address} returned an invalid score '{score}' ({type(score).__name__}), setting to 0.")
                    score = 0.0

            logger.debug(f"Dispatcher {dispatchers[index][0].address} scored {score}.")
            if score > highest_score:
                highest_score = score
                highest_score_index = index

        if highest_score_index == -1:
            # TODO: handle need for more servers (maybe)
            raise ValueError("No dispatcher is available to handle your request. Please try again later.")

        # Send prepare request to highest scoring dispatcher
        # Pass-through the response to the client
        dispatcher, resolve = dispatchers[highest_score_index]
        logger.debug(f"Issuing prepare command to dispatcher {dispatcher.address}.")
        executor_address_payload = await dispatcher(
            self.pack_control_message("prepare", request),
            timeout=self.dispatcher_prepare_timeout,
        )

        if resolve and not is_absolute_address(executor_address_payload["address"]):
            # Resolve relative address to dispatcher address
            logger.debug(f"Resolving relative address {executor_address_payload['address']} to dispatcher address.")
            executor_address_payload["address"] = get_absolute_address_from_relative(
                absolute_address=dispatcher.address,
                relative_address=executor_address_payload["address"],
            )

        return executor_address_payload
