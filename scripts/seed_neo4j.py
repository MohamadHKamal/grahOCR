"""Seed Neo4j with domain rules from config/neo4j_rules.yaml."""

import asyncio

from graphocr.core.config import get_settings
from graphocr.core.logging import setup_logging
from graphocr.layer2_verification.knowledge_graph.client import Neo4jClient
from graphocr.layer2_verification.knowledge_graph.schema_loader import load_schema


async def main() -> None:
    setup_logging("INFO")

    client = Neo4jClient()
    await client.connect()

    try:
        await load_schema(client)
        print("Neo4j schema and rules seeded successfully.")

        # Verify
        result = await client.execute_read("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count")
        for row in result:
            print(f"  {row['label']}: {row['count']} nodes")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
