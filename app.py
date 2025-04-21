import pydantic.main
from pydantic import BaseModel as PydanticBaseModel
pydantic.main.ModelMetaclass = PydanticBaseModel.__class__


from fastapi import FastAPI
# Ensure compatibility between Pydantic v1 and v2
try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel
from typing import Any, List, Optional
from pathlib import Path

# Import the existing Cog Predictor and ModelOutput from the 'sample' package
from sample.predict import Predictor, ModelOutput

# Define request and response schemas
def convert_path_list(paths: Optional[List[Path]]) -> Optional[List[str]]:
    if paths is None:
        return None
    return [str(p) for p in paths]

class PredictRequest(BaseModel):
    prompt: str
    num_repetitions: int = 3
    output_format: str = "animation"

class PredictResponse(BaseModel):
    json_file: Optional[Any]
    animation: Optional[List[str]]

# Create FastAPI app
app = FastAPI(
    title="Text-to-Motion API",
    description="A FastAPI wrapper around the Cog Predictor for generating 3D motions from text.",
    version="1.0.0"
)

# Initialize the predictor
predictor = Predictor()

@app.on_event("startup")
async def on_startup():
    # Load model, diffusion, and cache CLIP weights
    predictor.setup()

@app.post("/predict", response_model=PredictResponse)
def predict_motion(req: PredictRequest):
    # Run inference
    output: ModelOutput = predictor.predict(
        prompt=req.prompt,
        num_repetitions=req.num_repetitions,
        output_format=req.output_format
    )
    # Prepare response
    if req.output_format == "animation":
        return PredictResponse(
            animation=convert_path_list(output.animation),
            json_file=None
        )
    else:
        return PredictResponse(
            animation=None,
            json_file=output.json_file
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=2199, reload=True)

