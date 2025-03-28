
import os
import torchaudio
from create_explanations import read_file_line_by_line

def init_explaination_with_audio(text_file: str, audio_dir_path: str):
    explainations = read_file_line_by_line(text_file)
    files = [f for f in os.listdir(audio_dir_path) if os.path.isfile(os.path.join(audio_dir_path, f))]
    sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))
    for explaination, audio_file in zip(explainations, sorted_files):
        audio_path = os.path.join(audio_dir_path, audio_file)
        explaination.audio_path = audio_path
        audio_tensor = torchaudio.load(audio_path)
        explaination.audio_duration = audio_tensor[0].shape[1] / audio_tensor[1]
        explaination.audio_tensor = audio_tensor[0]
        
    return explainations


if __name__ == "__main__":
    text_file = "data/text/shenyang.txt"
    audio_dir_path = "data/audio/"
    init_explaination_with_audio(text_file, audio_dir_path)
