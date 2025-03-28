import tempfile
import logging
import sys
import os
import time

import torch
import torchaudio
import soundfile
import yaml

from pathlib import Path
from melo.api import TTS
from torchaudio.transforms import Resample
from starlette.responses import Response
from fastapi import FastAPI, APIRouter, BackgroundTasks, UploadFile, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import post_processing
from rag_llama_index import RagLlamaIndex
from utils import check_request_audio_file_path
from init_chat_resource import init_explaination_with_audio
from preprocessing_user_input import extract_int_from_string
from post_processing import text_post_processing


logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s | %(pathname)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format for timestamp
    stream=sys.stdout
)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# create configurations from the yaml file
conf = {}
with open("src/conf_http_server.yaml") as file:
    conf = yaml.safe_load(file)["conf_http_server"]
stt_model_path = conf["stt_model_path"]
embedding_model_path = conf["embedding_model_path"]
ollama_url = conf["ollama_url"]
text_path = conf["text_path"]
dir_path = conf["dir_path"]
audio_path = conf["audio_path"]
tts_speed = conf["tts_speed"]
response_audio_dir = conf["response_audio_dir"]
device = conf["device"]
delete_audio_file_after_seconds = 120
llm_mode = conf["llm_mode"]

explainations = None
rag_engine = None
tts_model, tts_speaker_ids, speed = None, None, tts_speed
stt_pipe = None


def initialize_fn():
    """Initialize corresponding AI models from configurations"""
    global explainations, rag_engine, tts_model, tts_speaker_ids, stt_pipe

    # pre-generated explainations
    explainations = init_explaination_with_audio(text_path,
                                                    audio_path)
    os.system(f"cp -r data/audio/ {response_audio_dir}")
    # RAG
    rag_engine = RagLlamaIndex(dir_path,
                                mode=llm_mode,
                                embedding_model=embedding_model_path,
                                ollama_url=ollama_url)

    # Text-To-Speech
    tts_model = TTS(language='ZH', device=device)
    tts_speaker_ids = tts_model.hps.data.spk2id

    # Speech-To-Text
    torch_dtype = torch.float16 if torch.cuda.is_available(
    ) else torch.float32
    stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        stt_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True)
    stt_model.to(device)
    stt_processor = AutoProcessor.from_pretrained(stt_model_path)
    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model=stt_model,
        tokenizer=stt_processor.tokenizer,
        feature_extractor=stt_processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )


def audio_to_text(audio_path: str) -> str:
    """Convert user audio to text
    Params:
        audio_path: str, path to the user audio, in wav format
    
    Returns:
        user_input: str, the text converted from the audio
    """
    assert audio_path is not None
    waveform, sample_rate = torchaudio.load(audio_path)
    # 1. merge multi channel audio
    if waveform.size(0) > 1:  # More than one channel
        # Average the channels to convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # 2. resample to 16 kHz
    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    waveform = waveform.squeeze().numpy()
    result = stt_pipe(waveform,
                        return_timestamps=True,
                        generate_kwargs={
                            "language": "chinese",
                        })
    user_input = result["text"]
    logging.info(f"stt: {user_input}")
    return user_input


def chat_and_response(input_text: str) -> Response:
    """Chat with the user and return the response
    Params:
        input_text: str, the user input text
    
    Returns:
        data: dict, the http response, including the text answer and audio file path
    """
    assert input_text is not None
    user_input = input_text

    # 1. Check if the input text contains number and directly return the corresponding explaination
    num = extract_int_from_string(user_input)
    if num is not None:
        if num <= 0 and num >= len(explainations):
            hint_msg = post_processing.DEFAULT_USER_PROMPT
            data = {
                "response": hint_msg,
                "audio_file": ""
            }
            return JSONResponse(content=data, media_type="application/json; charset=utf-8")
        explain = explainations[num]
        data = {
                "response": explain.text,
                "audio_file": os.path.join(os.getcwd(), response_audio_dir, f"audio/{num}.wav")
            }
        return JSONResponse(content=data, media_type="application/json; charset=utf-8")

    # 2. We query the RAG engine
    response = rag_engine.query(user_input).response
    logging.info(f"RAG response: {response}")
    response = text_post_processing(response)
    
    # 3. If the response is not None, we return the response and corresponding audio
    if response is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=response_audio_dir) as f:
            output_audio = tts_model.tts_to_file(response,
                                                    tts_speaker_ids['ZH'],
                                                    speed=speed)
            soundfile.write(f.name, output_audio,
                            tts_model.hps.data.sampling_rate)
            logging.info(f"Generated audio file: {f.name}")
            response_audio_file = os.path.abspath(f.name)
            data = {
                    "response": response,
                    "audio_file": response_audio_file
            }
            
            return JSONResponse(content=data, media_type="application/json; charset=utf-8")
    else:
        data = {
                "response": hint_msg,
                "audio_file": ""
        }
        return JSONResponse(content=data, media_type="application/json; charset=utf-8")


initialize_fn()


def audio_to_text(file: UploadFile) -> str:
    """Receive the uploaded the audio file and return the response"""
    uploaded_audio_path = os.path.abspath(file.filename)
    logging.info(f"Uploaded audio file: {uploaded_audio_path}")
    waveform, sample_rate = torchaudio.load(file.file)
    # 1. merge multi channel audio
    if waveform.size(0) > 1:  # More than one channel
        # Average the channels to convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # 2. resample to 16 kHz
    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    # 3. Convert the audio to text
    waveform = waveform.squeeze().numpy()
    result = stt_pipe(waveform,
                        return_timestamps=True,
                        generate_kwargs={
                            "language": "chinese",
                        })
    return result["text"]



def remove_file(path: str):
    time.sleep(delete_audio_file_after_seconds)
    if os.path.exists(path):
        os.remove(path)


app = FastAPI()
router = APIRouter()
api_key_header = APIKeyHeader(name="API-Key")


def check_api_key(api_key_header: str = Security(api_key_header)):
    api_key_env = os.getenv("API_KEY")
    if api_key_env is None:
        raise EnvironmentError("API_KEY not set in the environment")
    logging.info(f"API key: {api_key_header}, env key {api_key_env}")
    if api_key_env == api_key_header: 
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key"
    )


@router.post("/stt/")
async def service_audio_to_text(file: UploadFile, validate: bool = Depends(check_api_key)) -> Response:
    """Service Speech to Text: Get uploaded audio file and return the text"""
    if validate:
        text = audio_to_text(file)
        return JSONResponse(content={"response": text}, media_type="application/json; charset=utf-8")
    else:
        return JSONResponse(content={"error": "Missing or invalid API key"}, media_type="application/json; charset=utf-8")


@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile) -> Response:
    """Service Speech to Chat and Response: Receive the uploaded the audio file and return the chatbot response"""
    
    return chat_and_response(audio_to_text(file))


@router.get("/chat")
async def inference_fn(
    input_text: str | None,
    validate: bool = Depends(check_api_key)
):
    """Service Chat: Get the user input text and return the chatbot response"""
    if validate:
        return chat_and_response(input_text)
    else:
        return JSONResponse(content={"error": "Missing or invalid API key"}, media_type="application/json; charset=utf-8")


@router.get("/audios/")
async def get_audio(audio_file_path: str, background_tasks: BackgroundTasks):
    """Service get the audio file that is generated by the chatbot"""
    # Check whether the file path is valid to avoid directory traversal attacks
    if check_request_audio_file_path(audio_file_path, response_audio_dir) is False:
        return JSONResponse(content={"error": "File not found"}, media_type="application/json; charset=utf-8")
    if os.path.exists(audio_file_path):
        if audio_file_path.find(response_audio_dir):
            # Delete the file after certain seconds
            last_dir = Path(audio_file_path).parent.name
            # Here we don't delete the audio file if it is in the `audio` folder
            if delete_audio_file_after_seconds > 0 and last_dir != "audio":
                background_tasks.add_task(remove_file, audio_file_path)
        return FileResponse(audio_file_path, media_type="audio/mpeg", filename=f"{audio_file_path}_audio.wav")
    return JSONResponse(content={"error": "File not found"}, media_type="application/json; charset=utf-8")


app.include_router(
    router,
    prefix="/api",
    dependencies=[Depends(check_api_key)]
)

app.include_router(
    router,
    prefix="/api",
    dependencies=[Depends(check_api_key)]
)

app.include_router(
    router,
    prefix="/api",
    dependencies=[Depends(check_api_key)]
)

app.include_router(
    router,
    prefix="/api",
    dependencies=[Depends(check_api_key)]
)
