import json
import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app

def generate_openapi_spec():
    """Generates static openapi.json file from FastAPI application schema."""
    openapi_schema = app.openapi()
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "openapi.json"))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"Successfully generated OpenAPI specification file at: {output_path}")

if __name__ == "__main__":
    generate_openapi_spec()
