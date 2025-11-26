import os
import asyncio
import httpx
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP
import signal
import platform
import logging
import sys
from dotenv import load_dotenv
import base64
import requests
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.foundation_models import ModelInference


# Load variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize MCP server
mcp = FastMCP("Google Shopping Reviews Server")

# Configuration
SERPAPI_BASE_URL = "https://serpapi.com/search.json"
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Timeouts
REQUEST_TIMEOUT = 60.0
CONNECT_TIMEOUT = 60.0
DEFAULT_MIN_RATING = 1.0
MAX_SPEC_FEATURES = 15


async def api_call(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Make API call to SerpAPI"""
    timeout_config = httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)

    try:
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.get(SERPAPI_BASE_URL, params=params)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"API error: {str(e)[:100]}")
        return None
    
def extract_specs(specs_data: dict) -> List[dict]:
    """Recursively extract specifications from a dict"""
    specs = []
    for category, values in specs_data.items():
        if isinstance(values, dict):
            for name, value in values.items():
                specs.append({
                    "category": str(category),
                    "name": str(name),
                    "value": str(value)[:200]
                })
        else:
            specs.append({
                "category": str(category),
                "name": "value",
                "value": str(values)[:200]
            })
    return specs

def extract_specs_from_product(pr: dict) -> List[dict]:
    """Extract specifications from a product result, handling dicts, lists, or strings."""
    specs = []

    # 1️⃣ Try specs_results first
    specs_results = pr.get("specs_results", {})
    if specs_results:
        specs.extend(extract_specs(specs_results))

    # 2️⃣ Fallback: about_the_product.features
    about = pr.get("about_the_product", {})
    features = about.get("features", [])
    if features:
        if isinstance(features, dict):
            for k, v in list(features.items())[:MAX_SPEC_FEATURES]:
                specs.append({"name": str(k), "value": str(v)[:200]})
        elif isinstance(features, list):
            for i, val in enumerate(features[:MAX_SPEC_FEATURES]):
                specs.append({"name": f"feature_{i+1}", "value": str(val)[:200]})
        else:
            specs.append({"name": "feature", "value": str(features)[:200]})

    return specs

def encode_image_to_base64(image_url: str) -> Optional[str]:
    """
    Download an image from URL and encode it to base64.
    
    Args:
        image_url: The URL of the image to download
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()
        image_bytes = response.content
        encoded = base64.b64encode(image_bytes).decode('utf-8')
        return encoded
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        raise

@mcp.tool()
def extract_from_image(image_url: str) -> str:
    """
    Takes an image URL, encodes it to base64, and returns the name of the object 
    in the image using Watsonx.ai vision model.

    Args:
        image_url: The URL of the image file to analyze

    Returns:
        Generated description of the object in the image, including brand and model if available
    """
    
    # Get credentials from environment variables
    model_id = os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-3-2-11b-vision-instruct")
    api_key = os.getenv("WATSONX_APIKEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    
    if not api_key:
        raise ValueError("WATSONX_APIKEY environment variable is required")
    if not project_id:
        raise ValueError("WATSONX_PROJECT_ID environment variable is required")

    try:
        # Set up Watson credentials
        credentials = Credentials(
            url="https://us-south.ml.cloud.ibm.com", 
            api_key=api_key
        )

        # Download and encode the image
        base64_image = encode_image_to_base64(image_url)

        # Define model parameters
        params = TextChatParameters(
            temperature=1
        )

        # Initialize the model
        model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            params=params
        )

        # Prepare the question and messages
        question = ("Your job is to recognize the object in the image, as well as its brand and model, "
                   "if provided. Return a description in as few words as possible but always include "
                   "brand name and model.")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    }
                ]
            }
        ]

        # Generate response
        response = model.chat(messages=messages)
        
        # Extract and return the content
        result = response["choices"][0]["message"]["content"]
        logger.info(f"Successfully analyzed image: {image_url}")
        return result

    except Exception as e:
        logger.error(f"Error processing image {image_url}: {e}")
        raise RuntimeError(f"Failed to analyze image: {str(e)}")

@mcp.tool()
async def search_and_review_high_rated_products(
    query: str,
    min_rating: float = DEFAULT_MIN_RATING,
    max_products: int = 3,
    max_reviews_per_product: int = 5,
    location: str = "us",
    language: str = "en"
) -> Dict[str, Any]:
    """Search for products and retrieve reviews + specifications"""
    if not SERP_API_KEY:
        return {
            "success": False,
            "error": "SERP_API_KEY not configured",
            "products_with_reviews": []
        }

    try:
        # Step 1: Search for products
        search_params = {
            "engine": "google_shopping",
            "q": query,
            "api_key": SERP_API_KEY,
            "gl": location,
            "hl": language,
            "num": 20
        }
        search_data = await api_call(search_params)
        products = (
            search_data.get("shopping_results", [])
            or search_data.get("inline_shopping_results", [])
            or search_data.get("categorized_shopping_results", [{}])[0].get("shopping_results", [])
        )
        if not products:
            return {
                "success": False,
                "error": f"No products found for: {query}",
                "products_with_reviews": []
            }

        # Filter by rating
        if min_rating > 0:
            filtered = [p for p in products if not p.get("rating") or float(p.get("rating", 0)) >= min_rating]
            products = filtered if filtered else products[:max_products]

        results = []

        for product in products[:max_products]:
            product_id = product.get("product_id")
            result = {
                "basic_info": {
                    "title": product.get("title"),
                    "price": product.get("price"),
                    "rating": product.get("rating"),
                    "reviews_count": product.get("reviews"),
                    "source": product.get("source"),
                    "link": product.get("link") or product.get("product_link"),
                    "thumbnail": product.get("thumbnail"),
                    "product_id": product_id
                },
                "reviews": [],
                "specifications": []
            }

            # Prepare parallel API calls
            immersive_task = None
            specs_task = None

            token = product.get("immersive_product_page_token")
            if token:
                immersive_params = {
                    "engine": "google_immersive_product",
                    "page_token": token,
                    "api_key": SERP_API_KEY,
                    "gl": location,
                    "hl": language
                }
                immersive_task = api_call(immersive_params)

            if product_id:
                specs_params = {
                    "engine": "google_product",
                    "product_id": product_id,
                    "api_key": SERP_API_KEY,
                    "gl": location,
                    "hl": language,
                    "specs": "1"
                }
                specs_task = api_call(specs_params)

            immersive_data, product_data = await asyncio.gather(
                immersive_task if immersive_task else asyncio.sleep(0, result=None),
                specs_task if specs_task else asyncio.sleep(0, result=None)
            )

            # Immersive API reviews
            if immersive_data and immersive_data.get("product_results"):
                pr = immersive_data["product_results"]
                if pr.get("brand"):
                    result["basic_info"]["brand"] = pr["brand"]
                if pr.get("title"):
                    result["basic_info"]["full_title"] = pr["title"]

                user_reviews = pr.get("user_reviews", [])
                for review in user_reviews[:max_reviews_per_product]:
                    result["reviews"].append({
                        "rating": review.get("rating"),
                        "title": review.get("title"),
                        "text": review.get("text", "")[:500],
                        "user": review.get("user_name"),
                        "date": review.get("date"),
                        "source": review.get("source"),
                        "verified": review.get("verified", False)
                    })

            # Product API specs + reviews
            if product_data and product_data.get("specs_results"):
                pr = product_data["specs_results"]
                if not result["basic_info"].get("brand") and pr.get("brand"):
                    result["basic_info"]["brand"] = pr["brand"]
                result["specifications"].extend(extract_specs_from_product(pr))

                # Fallback reviews
                if not result["reviews"]:
                    user_reviews = pr.get("user_reviews", []) or pr.get("reviews", [])
                    for review in user_reviews[:max_reviews_per_product]:
                        if isinstance(review, dict):
                            result["reviews"].append({
                                "rating": review.get("rating"),
                                "title": review.get("title"),
                                "text": review.get("text", "")[:500],
                                "user": review.get("user_name") or review.get("user"),
                                "date": review.get("date"),
                                "source": review.get("source"),
                                "verified": review.get("verified", False)
                            })

            results.append(result)

        total_reviews = sum(len(r["reviews"]) for r in results)
        total_specs = sum(len(r["specifications"]) for r in results)

        return {
            "success": True,
            "query": query,
            "total_products_found": len(products),
            "products_returned": len(results),
            "total_reviews": total_reviews,
            "total_specifications": total_specs,
            "products_with_reviews": results
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Processing error: {str(e)[:200]}",
            "products_with_reviews": []
        }

# Main execution
def main():
    """Run the MCP server"""
    
    # Check for command line arguments
    port = int(os.getenv("MCP_PORT", 8080))
    host = os.getenv("MCP_HOST", "0.0.0.0")
    
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
        elif arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
    
    logger.info(f"Starting Shooping MCP Server on {host}:{port}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"SERP API Integration: {'Enabled' if SERP_API_KEY else 'Disabled'}")
    logger.info(f"Server Focus: Google Shooping API integration for product reviews")
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down MCP server...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    mcp.settings.port = port
    mcp.settings.host = host
    
    # Run with proper transport
    mcp.run(
         transport="sse"
    )
    

if __name__ == "__main__":
    main()