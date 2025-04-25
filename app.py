import pydantic.main
from pydantic import BaseModel as PydanticBaseModel
pydantic.main.ModelMetaclass = PydanticBaseModel.__class__
import subprocess
import json


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


@app.post("/generate", response_model=PredictResponse)
def generate_motion(req: PredictRequest):
    # Run the sample.generate script as a subprocess
    cmd = [
        "python", "-m", "sample.generate",
        "--model_path", "./save/humanml_mixamo_trans_enc_512/model000200000.pt",
        "--text_prompt", req.prompt,
        "--num_repetitions", str(1),
        "--num_samples", str(1),
        #"--output_format","json_file"
    ]


    import subprocess
    import json
    import os
    from datetime import datetime
    import numpy as np

    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Check if command was successful
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with error: {result.stderr}")

    # Get the latest model file in the specified directory
    model_dir = "save/humanml_mixamo_trans_enc_512"
    try:
        # Get all .npy files in the directory and subdirectories
        npy_files = []
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.npy'):
                    full_path = os.path.join(root, file)
                    # Get file creation time
                    file_time = datetime.fromtimestamp(os.path.getctime(full_path))
                    npy_files.append((full_path, file_time))

        # Sort by creation time (newest first)
        npy_files.sort(key=lambda x: x[1], reverse=True)

        # Get the latest .npy file path if any were found
        latest_npy = npy_files[0][0] if npy_files else None
        if latest_npy:
            # Load the latest .npy file
            with open(latest_npy, 'rb') as f:
                latest_model = np.load(f, allow_pickle=True)

            # Extract the 'motion' value from the loaded numpy file
            if 'motion' in latest_model.item():
                print("latest_model has motion ")
                return PredictResponse(
                    animation=None,
                    json_file={
                        'thetas':latest_model.item()['motion'].tolist(),
                        'root_translations':np.zeros((latest_model['motion'].shape[0], 3)).tolist(),
                    }
                )
            else:
                print("key error!!")
                raise KeyError("The loaded .npy file does not contain 'motion' key.")

        else:
            print("No .npy files found in the model directory")
    except Exception as e:
        print(f"Error finding latest model: {e}")
        return PredictResponse(
            animation=None,
            json_file={"error":str(e)}
        )


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
    uvicorn.run("app:app", host="0.0.0.0", port=8384, reload=True)

