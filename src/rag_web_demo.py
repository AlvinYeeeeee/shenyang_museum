import tempfile
import logging
import sys

from argparse import ArgumentParser

import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gradio as gr

from melo.api import TTS
import soundfile

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import post_processing
from rag_llama_index import RagLlamaIndex
from init_chat_resource import init_explaination_with_audio
from preprocessing_user_input import extract_int_from_string
from post_processing import text_post_processing

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="8888")
    parser.add_argument("--stt-model-path",
                        type=str,
                        default="whisper-large-v3-turbo")
    parser.add_argument("--embedding-model-path",
                        type=str,
                        default="BAAI/bge-large-zh-v1.5")
    parser.add_argument("--ollama-url",
                        type=str,
                        default="http://localhost:11434")
    parser.add_argument("--text-path",
                        type=str,
                        default="data/text/shenyang.txt")
    parser.add_argument("--dir-path", type=str, default="data/explanations")
    parser.add_argument("--audio-path", type=str, default="data/audio/")
    parser.add_argument("--tts-speed", type=float, default=1.0)

    args = parser.parse_args()

    device = "cuda:0"
    explainations = None
    rag_engine = None
    tts_model, tts_speaker_ids, speed = None, None, args.tts_speed
    stt_pipe = None

    def initialize_fn():
        global explainations, rag_engine, tts_model, tts_speaker_ids, stt_pipe

        # pre-generated explainations
        explainations = init_explaination_with_audio(args.text_path,
                                                     args.audio_path)

        # RAG
        rag_engine = RagLlamaIndex(args.dir_path,
                                   mode="ollama",
                                   embedding_model=args.embedding_model_path,
                                   ollama_url=args.ollama_url)

        # Text-To-Speech
        tts_model = TTS(language='ZH', device=device)
        tts_speaker_ids = tts_model.hps.data.spk2id

        # Speech-To-Text
        stt_model_path = args.stt_model_path
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

    def clear_fn():
        return [], [], '', '', '', None, None

    def inference_fn(
        temperature: float,
        top_p: float,
        max_new_token: int,
        input_mode,
        audio_path: str | None,
        input_text: str | None,
        history: list[dict],
        previous_input_tokens: str,
        previous_completion_tokens: str,
    ):
        # 1. If input audio, convert to text:
        if input_mode == "audio":
            assert audio_path is not None
            history.append({"role": "user", "content": {"path": audio_path}})
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
            history.append({"role": "user", "content": input_text})

        else:
            assert input_text is not None
            history.append({"role": "user", "content": input_text})
            user_input = input_text

        # 2. Check if the input text contains number
        num = extract_int_from_string(user_input)
        if num is not None and input_mode != "audio":
            if num <= 0 and num >= len(explainations):
                hint_msg = post_processing.DEFAULT_USER_PROMPT
                yield history + hint_msg, user_input, hint_msg, '', None, (
                    22050, explainations[0].audio_tensor.squeeze().numpy())
                return
            explain = explainations[num]
            history.append({
                "role": "assistant",
                "content": {
                    "path": explain.audio_path,
                    "type": "audio/wav"
                }
            })
            history.append({"role": "assistant", "content": explain.text})
            yield history, "", explain.text, '', None, (
                22050, explain.audio_tensor.squeeze().numpy())
            return

        # 3. We query the RAG engine
        response = rag_engine.query(user_input).response
        logging.info(f"RAG response: {response}")
        response = text_post_processing(response)

        # 4. If the response is not None, we return the response and corresponding audio
        if response is not None:
            # history.append({"role": "assistant", "content": response})
            # yield history, "", response, '', None, None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_audio = tts_model.tts_to_file(response,
                                                     tts_speaker_ids['ZH'],
                                                     speed=speed)
                soundfile.write(f.name, output_audio,
                                tts_model.hps.data.sampling_rate)
                logging.info(f"Generated audio file: {f.name}")
                history.append({"role": "assistant", "content": response})
                history.append({
                    "role": "assistant",
                    "content": {
                        "path": f.name,
                        "type": "audio/wav"
                    }
                })
            yield history, "", response, '', None, (
                tts_model.hps.data.sampling_rate, output_audio)
        # Gather history
        # inputs = previous_input_tokens + previous_completion_tokens
        # inputs = inputs.strip()
        else:
            yield history, "", "请重新提问", '', None, (
                tts_model.hps.data.sampling_rate, output_audio)

    def update_input_interface(input_mode):
        if input_mode == "audio":
            return [gr.update(visible=True), gr.update(visible=False)]
        else:
            return [gr.update(visible=False), gr.update(visible=True)]

    # Create the Gradio interface
    with gr.Blocks(title="RAG-IT Demo", fill_height=True) as demo:
        with gr.Row():
            temperature = gr.Number(label="Temperature", value=0.2)

            top_p = gr.Number(label="Top p", value=0.8)

            max_new_token = gr.Number(
                label="Max new tokens",
                value=2000,
            )

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            type="messages",
            scale=1,
        )

        with gr.Row():
            with gr.Column():
                input_mode = gr.Radio(["audio", "text"],
                                      label="Input Mode",
                                      value="audio")
                audio = gr.Audio(label="Input audio",
                                 type='filepath',
                                 show_download_button=True,
                                 visible=True)
                text_input = gr.Textbox(label="Input text",
                                        placeholder="Enter your text here...",
                                        lines=2,
                                        visible=False)

            with gr.Column():
                submit_btn = gr.Button("Submit")
                reset_btn = gr.Button("Clear")
                output_audio = gr.Audio(label="Play",
                                        streaming=True,
                                        autoplay=True,
                                        show_download_button=False)
                complete_audio = gr.Audio(label="Last Output Audio (If Any)",
                                          show_download_button=True)

        gr.Markdown("""## Debug Info""")
        with gr.Row():
            input_tokens = gr.Textbox(
                label=f"Input Tokens",
                interactive=False,
            )

            completion_tokens = gr.Textbox(
                label=f"Completion Tokens",
                interactive=False,
            )

        detailed_error = gr.Textbox(
            label=f"Detailed Error",
            interactive=False,
        )

        history_state = gr.State([])

        respond = submit_btn.click(inference_fn,
                                   inputs=[
                                       temperature,
                                       top_p,
                                       max_new_token,
                                       input_mode,
                                       audio,
                                       text_input,
                                       history_state,
                                       input_tokens,
                                       completion_tokens,
                                   ],
                                   outputs=[
                                       history_state, input_tokens,
                                       completion_tokens, detailed_error,
                                       output_audio, complete_audio
                                   ])

        respond.then(lambda s: s, [history_state], chatbot)

        reset_btn.click(clear_fn,
                        outputs=[
                            chatbot, history_state, input_tokens,
                            completion_tokens, detailed_error, output_audio,
                            complete_audio
                        ])
        input_mode.input(clear_fn,
                         outputs=[
                             chatbot, history_state, input_tokens,
                             completion_tokens, detailed_error, output_audio,
                             complete_audio
                         ]).then(update_input_interface,
                                 inputs=[input_mode],
                                 outputs=[audio, text_input])

    initialize_fn()
    # Launch the interface
    demo.launch(server_port=args.port, server_name=args.host, share=True)
