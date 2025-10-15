import json
import logging
from mangum import Mangum
from fastapi import FastAPI, Request

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = FastAPI()

@app.post("/evaluation_endpoint")
async def evaluation_endpoint(request: Request, data: dict = None):
    logger.info(f"✅ HIT /evaluation_endpoint")
    return {
        "status": "success",
        "message": "Successfully hit /evaluation_endpoint",
        "path": request.url.path,
        "data": data
    }

@app.post("/events")
async def events_endpoint(request: Request, data: dict = None):
    logger.info(f"❌ HIT /events (wrong endpoint)")
    return {
        "status": "wrong_endpoint",
        "message": "Hit /events instead of /evaluation_endpoint",
        "path": request.url.path,
        "data": data
    }

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    logger.info(f"🔍 CATCH-ALL: Received request to /{path}")
    return {
        "status": "catch_all",
        "message": f"Caught request to undefined path: /{path}",
        "method": request.method,
        "path": f"/{path}",
        "full_url": str(request.url)
    }

def lambda_handler(event, context):
    # Log everything we receive
    logger.info("=" * 60)
    logger.info("🔍 LAMBDA INVOCATION - FULL EVENT")
    logger.info("=" * 60)
    logger.info(json.dumps(event, indent=2))
    logger.info("=" * 60)
    
    # Log specific path fields
    logger.info(f"📍 path: {event.get('path')}")
    logger.info(f"📍 rawPath: {event.get('rawPath')}")
    logger.info(f"📍 resource: {event.get('resource')}")
    
    if 'requestContext' in event:
        logger.info(f"📍 requestContext.path: {event['requestContext'].get('path')}")
        if 'http' in event['requestContext']:
            logger.info(f"📍 requestContext.http.path: {event['requestContext']['http'].get('path')}")
    
    logger.info("=" * 60)
    
    # Process with Mangum
    handler = Mangum(app)
    response = handler(event, context)
    
    logger.info("📤 RESPONSE:")
    logger.info(json.dumps(response, indent=2))
    
    return response
