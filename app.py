import pydantic.main
from pydantic import BaseModel as PydanticBaseModel
pydantic.main.ModelMetaclass = PydanticBaseModel.__class__
import subprocess
import json
import os
import google.generativeai as genai

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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")
genai.configure(api_key=GOOGLE_API_KEY)

def refine_prompt(prompt: str) -> str:
    system_prompt = """
    당신은 텍스트를 3D 모션 생성에 적합한 형태로 변환하는 전문가입니다.
    입력된 텍스트를 다음 기준에 맞춰 정제해주세요:

    1. 동작의 속도, 강도, 감정을 명확하게 표현
    2. 신체의 각 부위(팔, 다리, 몸통 등)의 움직임을 구체적으로 기술
    3. 동작의 시작과 끝을 명확하게 정의
    4. 불필요한 수식어나 모호한 표현 제거

    예시:
    입력 텍스트: 활기차게 걷기
    변환된 텍스트: Walks energetically with arms swinging and chest lifted

    입력 텍스트: 슬프게 고개를 숙임
    변환된 텍스트: Slowly lowers head while shoulders droop to express sadness

    입력 텍스트: 화난 듯이 빠르게 달려듦
    변환된 텍스트: Charges forward rapidly with clenched fists and stiff posture

    입력 텍스트: 부끄럽게 손을 흔듦
    변환된 텍스트: Waves hand gently while looking down and shifting body weight shyly

    입력 텍스트: 기쁘게 점프하며 팔을 듦
    변환된 텍스트: Jumps upward with both arms raised high, expressing joy

    출력 형식:
    - 한 문장으로 된 명확한 동작 설명 (영어로만)
    """
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([
        system_prompt,
        f"입력 텍스트: {prompt}\n변환된 텍스트:"
    ])
    if response.text:
        refined_text = response.text.strip()
        # 한글 제거
        refined_text = ''.join([c for c in refined_text if not ('\u4e00' <= c <= '\u9fff' or '\u3130' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7AF')])
        return refined_text.strip()
    else:
        raise ValueError("프롬프트 정제 실패: API 응답이 비어있습니다.")

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
    # 프롬프트 정제 (이미 추가되어 있다고 가정)
    refined_prompt = refine_prompt(req.prompt)
    # Run inference with 정제된 프롬프트
    output: ModelOutput = predictor.predict(
        prompt=refined_prompt,
        num_repetitions=req.num_repetitions,
        output_format=req.output_format
    )
    # motions array만 반환
    motions = None
    if hasattr(output, "motions"):
        motions = output.motions
    elif hasattr(output, "json_file") and output.json_file and "motions" in output.json_file:
        motions = output.json_file["motions"]
    elif hasattr(output, "json_file") and output.json_file and "thetas" in output.json_file:
        motions = output.json_file["thetas"]
    else:
        motions = []
    return {"json_file": {"motions": motions}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8384, reload=True)

