"""
Batch-delete Neo4j edges written by the imputation pipeline (source='imputation').

Deletes :SupplyProductTo relationships where r.source = 'imputation' in batches of
--batch-size, so no single transaction ever holds millions of deletions (the
transaction heap stays bounded and other sessions keep working between batches).

Usage:
    python Prediction/delete_imputation_edges.py --dry-run          # count only, delete nothing
    python Prediction/delete_imputation_edges.py                    # interactive confirm, then delete
    python Prediction/delete_imputation_edges.py --yes              # skip the confirmation prompt
    python Prediction/delete_imputation_edges.py --model gatgru_vec # only delete edges of one model
    python Prediction/delete_imputation_edges.py --batch-size 20000 --relation SupplyProductTo
"""

import argparse
import sys
import os
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from Data.neo4j_SPLC import Neo4jClient


def count_edges(client: Neo4jClient, relation: str, model: str = '') -> int:
    where = "WHERE r.source = 'imputation'"
    if model:
        where += " AND r.model = $model"
    query = (
        f"MATCH ()-[r:{relation}]->() {where} "
        f"RETURN count(r) AS n"
    )
    params = {'model': model} if model else None
    records = client.execute_query(query, parameters=params)
    return records[0]['n'] if records else 0


def delete_batch(client: Neo4jClient, relation: str, batch_size: int,
                 model: str = '') -> int:
    """Delete up to batch_size imputation edges; return how many were deleted."""
    where = "WHERE r.source = 'imputation'"
    params = {'limit': batch_size}
    if model:
        where += " AND r.model = $model"
        params['model'] = model
    query = (
        f"MATCH ()-[r:{relation}]->() {where} "
        f"WITH r LIMIT $limit "
        f"DELETE r "
        f"RETURN count(r) AS deleted"
    )
    records = client.execute_query(query, parameters=params)
    return records[0]['deleted'] if records else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-delete Neo4j edges with source='imputation'.")
    parser.add_argument('--relation', default='SupplyProductTo',
                        help="Relationship type to scan (default: SupplyProductTo)")
    parser.add_argument('--model', default='',
                        help="Optionally restrict to one model name (r.model = <name>). "
                             "Empty = delete every source='imputation' edge.")
    parser.add_argument('--batch-size', type=int, default=10_000,
                        help="Edges deleted per transaction (default: 10000)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Only count matching edges, delete nothing")
    parser.add_argument('--yes', action='store_true',
                        help="Skip the interactive confirmation prompt")
    args = parser.parse_args()

    client = Neo4jClient()

    target = (f"edges with source='imputation' AND model='{args.model}'"
              if args.model else "ALL edges with source='imputation'")
    total = count_edges(client, args.relation, args.model)
    print(f"[Scan] {total:,} {target} on :{args.relation}")

    if args.dry_run:
        print("[Dry-run] Nothing deleted.")
        return
    if total == 0:
        print("Nothing to delete.")
        return

    if not args.yes:
        answer = input(f"Permanently DELETE {total:,} edges from Neo4j? Type 'yes' to confirm: ")
        if answer.strip().lower() != 'yes':
            print("Aborted.")
            return

    deleted_total = 0
    t0 = time.time()
    while True:
        deleted = delete_batch(client, args.relation, args.batch_size, args.model)
        deleted_total += deleted
        elapsed = time.time() - t0
        rate = deleted_total / elapsed if elapsed > 0 else 0.0
        remaining = max(total - deleted_total, 0)
        eta = remaining / rate if rate > 0 else float('inf')
        print(f"\r[Delete] {deleted_total:,}/{total:,} "
              f"({deleted_total / total * 100:5.1f}%)  "
              f"{rate:,.0f} edges/s  ETA {eta/60:4.1f} min   ", end='', flush=True)
        if deleted < args.batch_size:
            break

    # Final verification
    leftover = count_edges(client, args.relation, args.model)
    print(f"\n[Done] Deleted {deleted_total:,} edges in "
          f"{(time.time() - t0)/60:.1f} min; {leftover:,} matching edges remain.")


if __name__ == '__main__':
    main()
