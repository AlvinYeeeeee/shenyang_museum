import re
import argparse
import os
import torchaudio, torch
import pickle
from transformers import WhisperFeatureExtractor, AutoTokenizer
from model_server import ModelWorker
from flow_inference import AudioDecoder
from inference_utils import inference_fn


def initialize_fn(args):
    flow_config = os.path.join(args.flow_path, "config.yaml")
    flow_checkpoint = os.path.join(args.flow_path, 'flow.pt')
    hift_checkpoint = os.path.join(args.flow_path, 'hift.pt')
    # GLM
    glm_tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                  trust_remote_code=True)
    # Flow & Hift
    audio_decoder = AudioDecoder(config_path=flow_config,
                                 flow_ckpt_path=flow_checkpoint,
                                 hift_ckpt_path=hift_checkpoint,
                                 device=args.device)
    model = ModelWorker(args.model_path, args.dtype, args.device)
    return model, glm_tokenizer, audio_decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-path",
                        type=str,
                        default="./glm-4-voice-decoder")
    parser.add_argument("--model-path",
                        type=str,
                        default="THUDM/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path",
                        type=str,
                        default="THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--text-path",
                        type=str,
                        default="models/RAG/shenyang.txt")
    parser.add_argument("--output_dir",
                        type=str,
                        default="models/RAG/audios")

    args = parser.parse_args()

    explanations = read_file_line_by_line(args.text_path)

    model, glm_tokenizer, audio_decoder = initialize_fn(args)

    def generate_audio(explanation: Explanation):
        text, tts_speech, complete_text = inference_fn(
            temperature=1.0,
            top_p=1.0,
            max_new_token=2048,
            input_text=f"请完整的复述冒号后的内容，不要增加额外的内容：{explanation.text}",
            model=model,
            glm_tokenizer=glm_tokenizer,
            audio_decoder=audio_decoder,
            device=args.device)
        audio_path = os.path.join(args.output_dir, f"{explanation.serial_number}.wav")
        with open(audio_path, "wb") as f:
            torchaudio.save(f, tts_speech.unsqueeze(0), 22050, format="wav")
        
        explanation.audio_path = audio_path
        explanation.audio_duration = torchaudio.info(audio_path).num_frames / 22050
        return explanation

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    for idx, explanation in enumerate(explanations):
        print(explanation)
        print("-" * 50)
        explanations[idx] = generate_audio(explanation)

    pickle.dump(explanations, open("explanations.pkl", "wb"))
    