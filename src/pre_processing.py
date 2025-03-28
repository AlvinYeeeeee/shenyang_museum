

import torchaudio
from torchaudio.transforms import Resample
import logging


def audio_pre_processing(input_wav, output_wav):
    waveform, sample_rate = torchaudio.load(input_wav)
    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    torchaudio.save(output_wav, waveform, 16000)
    logging.info(f"Converted {input_wav} from {sample_rate} Hz to {output_wav} at 16 kHz")