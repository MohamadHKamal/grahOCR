"""Async Neo4j driver wrapper with connection pooling."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from graphocr.core.config import get_settings
from graphocr.core.logging import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """Async Neo4j client with connection pooling and health checks."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        settings = get_settings()
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Initialize the driver connection pool."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
            max_connection_pool_size=50,
        )
        await self._driver.verify_connectivity()
        logger.info("neo4j_connected", uri=self._uri)

    async def close(self) -> None:
        """Close the driver and all connections."""
        if self._driver:
            await self._driver.close()
            logger.info("neo4j_disconnected")

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4j client not connected. Call connect() first.")
        return self._driver

    def session(self, database: str = "neo4j") -> AsyncSession:
        return self.driver.session(database=database)

    async def execute_read(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        """Execute a read query and return results as dicts."""
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def execute_write(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        """Execute a write query and return results as dicts."""
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def health_check(self) -> bool:
        """Check if Neo4j is reachable."""
        try:
            await self.execute_read("RETURN 1 AS ok")
            return True
        except Exception:
            return False
