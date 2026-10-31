import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from maintaina.config import KNOWLEDGE_DIR, MODEL_DIR
from maintaina.predict import ModelNotReady, PredictionService
from maintaina.rag import KnowledgeBase, build_query, guidance
from maintaina.schemas import (Guidance, GuidanceRequest, HealthResponse, PredictionResponse, SensorReading)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("maintaina.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = PredictionService(app.state.model_dir)
    kb = KnowledgeBase()
    kb.ingest_dir(app.state.knowledge_dir)
    app.state.kb = kb
    yield


def create_app(model_dir=MODEL_DIR, knowledge_dir=KNOWLEDGE_DIR) -> FastAPI:
    app = FastAPI(title="Maintaina AI", version="1.0.0", lifespan=lifespan,
                  description="Failure-risk estimate, anomaly flag and maintenance-manual retrieval. "
                              "Decision support only.")
    app.state.model_dir, app.state.knowledge_dir = model_dir, knowledge_dir

    @app.exception_handler(ModelNotReady)
    async def not_ready(_: Request, exc: ModelNotReady):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request):
        svc, kb = request.app.state.service, request.app.state.kb
        return HealthResponse(status="ok" if svc.ready else "degraded", model_loaded=svc.ready,
                              knowledge_chunks=len(kb.chunks))

    @app.get("/model-info", tags=["system"])
    def model_info(request: Request):
        svc = request.app.state.service
        if not svc.ready:
            raise ModelNotReady("Model artifacts are not available. Train the model first.")
        return svc.meta

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(reading: SensorReading, request: Request, include_guidance: bool = True):
        svc, kb = request.app.state.service, request.app.state.kb
        result = svc.predict(reading.model_dump())
        if include_guidance and (result["alert"] or result["anomaly"]):
            q = build_query(result["unusual_inputs"], result["alert"], result["anomaly"])
            result["guidance"] = guidance(kb, q)
        return result

    @app.post("/guidance", response_model=Guidance, tags=["rag"])
    def search_guidance(body: GuidanceRequest, request: Request):
        return guidance(request.app.state.kb, body.query, body.k)

    @app.post("/reindex", tags=["rag"])
    def reindex(request: Request):
        n = request.app.state.kb.ingest_dir(request.app.state.knowledge_dir)
        return {"knowledge_chunks": n}

    return app


app = create_app()
