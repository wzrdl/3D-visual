import os
import json

from app.backend_client import BackendAPIClient


def main() -> None:
    
    api_url = os.getenv("BACKEND_API_URL")
    print(f"Using backend: {api_url!r}")

    client = BackendAPIClient(api_url=api_url)
    try:
        models = client.list_models()
    finally:
        client.close()

    print(f"\nTotal models in backend DB: {len(models)}\n")

    for i, m in enumerate(models, start=1):
        print(f"--- Model #{i} ---")
        print(json.dumps(m, ensure_ascii=False, indent=2))
        print()

if __name__ == "__main__":
    main()