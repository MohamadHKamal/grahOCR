"""Ingest insurance policy documents into the vector store.

Usage:
    python scripts/ingest_policies.py --policy-dir ./data/policies/
    python scripts/ingest_policies.py --policy-file policy.json
"""

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

from graphocr.core.logging import setup_logging, get_logger
from graphocr.models.policy import CoverageRule, PolicyDocument, PolicyStatus, PolicyType
from graphocr.rag.policy_chunker import chunk_policy
from graphocr.rag.vector_store import PolicyVectorStore

logger = get_logger(__name__)


def load_policy_from_json(path: Path) -> PolicyDocument:
    """Load a policy document from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))

    coverage_rules = []
    for rule_data in data.get("coverage_rules", []):
        coverage_rules.append(CoverageRule(**rule_data))

    return PolicyDocument(
        policy_number=data["policy_number"],
        policy_type=PolicyType(data.get("policy_type", "standard")),
        status=PolicyStatus(data.get("status", "active")),
        effective_date=date.fromisoformat(data["effective_date"]),
        expiry_date=date.fromisoformat(data["expiry_date"]) if data.get("expiry_date") else None,
        version=data.get("version", "1.0"),
        supersedes=data.get("supersedes"),
        insurer_name=data.get("insurer_name", ""),
        plan_name=data.get("plan_name", ""),
        plan_name_ar=data.get("plan_name_ar", ""),
        jurisdiction=data.get("jurisdiction", ""),
        coverage_rules=coverage_rules,
        general_exclusions=data.get("general_exclusions", []),
        full_text=data.get("full_text", ""),
        full_text_ar=data.get("full_text_ar", ""),
        parent_policy_id=data.get("parent_policy_id"),
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest policies into vector store")
    parser.add_argument("--policy-dir", type=str, help="Directory of policy JSON files")
    parser.add_argument("--policy-file", type=str, help="Single policy JSON file")
    parser.add_argument("--store-dir", type=str, default="./data/vectorstore")
    args = parser.parse_args()

    setup_logging("INFO")
    store = PolicyVectorStore(persist_dir=args.store_dir)

    files: list[Path] = []
    if args.policy_dir:
        files = list(Path(args.policy_dir).glob("*.json"))
    elif args.policy_file:
        files = [Path(args.policy_file)]
    else:
        print("Provide --policy-dir or --policy-file")
        return

    total_chunks = 0
    for f in files:
        print(f"Processing: {f.name}")
        policy = load_policy_from_json(f)
        chunks = chunk_policy(policy)
        count = store.add_chunks(chunks)
        total_chunks += count
        print(f"  -> {count} chunks indexed (policy {policy.policy_number} v{policy.version})")

    print(f"\nTotal: {total_chunks} chunks from {len(files)} policies")
    print(f"Vector store: {store.count} total chunks")


if __name__ == "__main__":
    main()
