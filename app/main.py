from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.integrations.wecom.routes import router as wecom_router
from app.middleware import setup_request_logging


WEB_DIR = Path(__file__).parent / "web" / "static"

app = FastAPI(title="RAG Knowledge QA System")
setup_request_logging(app)
app.include_router(router)
app.include_router(wecom_router)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")
