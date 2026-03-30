"""Ensure Neo4j and ChromaDB are seeded — safe to run multiple times.

Usage:
    python scripts/setup_dbs.py          # seed both
    python scripts/setup_dbs.py --neo4j  # seed Neo4j only
    python scripts/setup_dbs.py --chroma # seed ChromaDB only
"""

import argparse
import asyncio
import sys
from pathlib import Path


async def seed_neo4j() -> bool:
    """Seed Neo4j if reachable and empty (or force-refresh)."""
    try:
        from graphocr.layer2_verification.knowledge_graph.client import Neo4jClient
        from graphocr.layer2_verification.knowledge_graph.schema_loader import load_schema
    except ImportError:
        print("[neo4j] SKIP — neo4j driver not installed")
        return False

    client = Neo4jClient()
    try:
        await client.connect()
    except Exception as e:
        print(f"[neo4j] SKIP — cannot connect ({e})")
        print("[neo4j] Make sure Docker is running: docker compose up -d")
        return False

    try:
        result = await client.execute_read(
            "MATCH (n) RETURN count(n) AS total"
        )
        count = result[0]["total"] if result else 0

        if count > 0:
            print(f"[neo4j] Already seeded ({count} nodes) — skipping")
            return True

        await load_schema(client)
        result = await client.execute_read(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt"
        )
        for row in result:
            print(f"  {row['label']}: {row['cnt']} nodes")
        print("[neo4j] Seeded successfully")
        return True
    finally:
        await client.close()


def seed_chroma(store_dir: str = "./data/vectorstore", policy_dir: str = "tests/fixtures/sample_policies") -> bool:
    """Seed ChromaDB if the vector store is empty."""
    store_path = Path(store_dir)
    policy_path = Path(policy_dir)

    if not policy_path.exists():
        print(f"[chroma] SKIP — policy dir not found: {policy_path}")
        return False

    try:
        from graphocr.rag.vector_store import PolicyVectorStore
        from graphocr.rag.policy_chunker import chunk_policy
        from scripts.ingest_policies import load_policy_from_json
    except ImportError as e:
        print(f"[chroma] SKIP — missing dependency ({e})")
        return False

    store = PolicyVectorStore(persist_dir=store_dir)

    if store.count > 0:
        print(f"[chroma] Already seeded ({store.count} chunks) — skipping")
        return True

    files = list(policy_path.glob("*.json"))
    if not files:
        print(f"[chroma] SKIP — no JSON files in {policy_path}")
        return False

    total_chunks = 0
    for f in files:
        policy = load_policy_from_json(f)
        chunks = chunk_policy(policy)
        count = store.add_chunks(chunks)
        total_chunks += count
        print(f"  {f.name}: {count} chunks (policy {policy.policy_number})")

    print(f"[chroma] Seeded {total_chunks} chunks from {len(files)} policies")
    return True


async def main():
    parser = argparse.ArgumentParser(description="Ensure Neo4j and ChromaDB are seeded")
    parser.add_argument("--neo4j", action="store_true", help="Seed Neo4j only")
    parser.add_argument("--chroma", action="store_true", help="Seed ChromaDB only")
    args = parser.parse_args()

    # If neither flag is set, seed both
    do_neo4j = args.neo4j or (not args.neo4j and not args.chroma)
    do_chroma = args.chroma or (not args.neo4j and not args.chroma)

    ok = True
    if do_neo4j:
        print("=== Neo4j ===")
        ok = await seed_neo4j() and ok

    if do_chroma:
        print("=== ChromaDB ===")
        ok = seed_chroma() and ok

    if ok:
        print("\nAll databases ready.")
    else:
        print("\nSome databases were skipped — see messages above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
