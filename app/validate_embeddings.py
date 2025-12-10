"""
Quick validation for the semantic search pipeline (sentence-transformers + cosine similarity).

Usage examples:
    python -m app.validate_embeddings --query "tree" --top-k 10
    python -m app.validate_embeddings --query "car"
"""

import argparse
import os

from app.client_data_manager import ClientDataManager


def main():
    parser = argparse.ArgumentParser(description="Validate semantic search (sentence-transformers)")
    parser.add_argument("--query", type=str, default="tree", help="Query text")
    parser.add_argument("--top-k", type=int, default=20, dest="top_k", help="Number of results to display")
    parser.add_argument("--backend-url", type=str, default=None, help="Override BACKEND_API_URL")
    parser.add_argument("--debug", action="store_true", help="Print debug info for the query")
    args = parser.parse_args()

    if args.backend_url:
        os.environ["BACKEND_API_URL"] = args.backend_url

    dm = ClientDataManager(api_url=args.backend_url)
    try:
        if args.debug:
            models = dm.get_all_models()
            print(f"Models count: {len(models)}")
            names_with_query = [
                m.get("display_name") for m in models
                if args.query.lower() in str(m.get("display_name", "")).lower()
            ]
            print(f"Names containing '{args.query}':", names_with_query[:20])
            
            if dm._encoder is not None:
                print(f"Encoder model: {dm._semantic_model_name}")
                print(f"Embedding dimensions: {dm._semantic_embeddings.shape if dm._semantic_embeddings is not None else 'None'}")
            else:
                print("Semantic encoder not initialized")

        results = dm.semantic_search(args.query, top_k=args.top_k)
        if not results:
            print("No embeddings were built; ensure backend metadata is reachable.")
            return

        print(f'Query="{args.query}" -> top {len(results)} matches')
        for item in results:
            model = item["model"]
            score = item["score"]
            tags = model.get("tags", [])
            print(f"{score:0.3f}\t{model.get('id')}\t{model.get('display_name')}\ttags={tags}")
    finally:
        dm.close()


if __name__ == "__main__":
    main()

