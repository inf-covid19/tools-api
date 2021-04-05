from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tools import api

app = FastAPI(
    title="COVID-19 Analysis Tools API",
    description="API for https://covid19.ufrgs.dev/tools",
)


@app.get("/_ah/health", tags=["system"])
def health():
    return {"is_healthy": True}


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    api.router,
    prefix="/api/v1",
)
