from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SensorReading(BaseModel):
    """One snapshot of machine sensors. Bounds are loose physical sanity limits, not training ranges."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "type": "M", "air_temperature": 300.0, "process_temperature": 310.0,
        "rotational_speed": 1500, "torque": 40.0, "tool_wear": 100}})

    type: Literal["L", "M", "H"] = Field(description="Product quality variant")
    air_temperature: float = Field(ge=250, le=400, description="Air temperature in kelvin")
    process_temperature: float = Field(ge=250, le=450, description="Process temperature in kelvin")
    rotational_speed: float = Field(gt=0, le=10000, description="Rotational speed in rpm")
    torque: float = Field(ge=0, le=500, description="Torque in Nm")
    tool_wear: float = Field(ge=0, le=1000, description="Tool wear in minutes")


class UnusualInput(BaseModel):
    feature: str
    value: float
    z_score: float = Field(description="Distance from the training mean in standard deviations")


class Passage(BaseModel):
    text: str
    source: str
    page: int | None = None
    score: float = Field(description="Retrieval similarity (0-1). Not a confidence in correctness.")


class Guidance(BaseModel):
    query: str
    passages: list[Passage]
    message: str


class PredictionResponse(BaseModel):
    failure_probability: float = Field(ge=0, le=1, description="Model score; uncalibrated estimate")
    alert: bool
    threshold: float
    anomaly: bool
    anomaly_score: float = Field(description="Lower = more unusual. NOT a failure probability")
    unusual_inputs: list[UnusualInput]
    warnings: list[str]
    guidance: Guidance | None = None
    disclaimer: str


class GuidanceRequest(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    k: int = Field(default=3, ge=1, le=10)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    knowledge_chunks: int
