import random
import pydantic.main
from pydantic import BaseModel as PydanticBaseModel
pydantic.main.ModelMetaclass = PydanticBaseModel.__class__
import subprocess
import json
import os
import google.generativeai as genai

from fastapi import FastAPI
# Ensure compatibility between Pydantic v1 and v2
from pydantic import BaseModel
from typing import Any, List, Optional
from pathlib import Path

# Import the existing Cog Predictor and ModelOutput from the 'sample' package
from sample.predict import Predictor, ModelOutput
from dotenv import load_dotenv

load_dotenv()

# Define request and response schemas
def convert_path_list(paths: Optional[List[Path]]) -> Optional[List[str]]:
    if paths is None:
        return None
    return [str(p) for p in paths]

class PredictRequest(BaseModel):
    prompt: str
    num_repetitions: int
    output_format: str

class PredictResponse(BaseModel):
    json_file: Optional[Any]
    animation: Optional[Any]

# Create FastAPI app
app = FastAPI(
    title="Text-to-Motion API",
    description="A FastAPI wrapper around the Cog Predictor for generating 3D motions from text.",
    version="1.0.0"
)

# Initialize the Predictor
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
    변환된 텍스트: a person walks energetically with arms swinging and chest lifted

    입력 텍스트: 슬프게 고개를 숙임
    변환된 텍스트: a person slowly lowers head while shoulders droop to express sadness

    입력 텍스트: 화난 듯이 빠르게 달려듦
    변환된 텍스트: a person charges forward rapidly with clenched fists and stiff posture

    입력 텍스트: 부끄럽게 손을 흔듦
    변환된 텍스트: a person waves hand gently while looking down and shifting body weight shyly

    입력 텍스트: 기쁘게 점프하며 팔을 듦
    변환된 텍스트: a person jumps upward with both arms raised high, expressing joy

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
    print("Loading model and diffusion...")
    predictor.setup()



from fastapi import Body

@app.post("/predict")
def predict_motion(req: PredictRequest = Body(...)):
    print(f"prompt = {req.prompt}, num_repetitions = {req.num_repetitions}, output_format = {req.output_format}")
    # 프롬프트 정제 (이미 추가되어 있다고 가정)
    refined_prompt = refine_prompt(req.prompt)
    print(f"refined_prompt = {refined_prompt}")
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

    import numpy as np
    # motions가 numpy array가 아니면 변환
    motions_np = np.array(motions)
    shape = list(motions_np.shape)
    framecount = shape[0] if len(shape) > 0 else 0

    return {"json_file": {"motions": motions, "shape": shape, "framecount": framecount}}  

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("API_PORT"), reload=True)

