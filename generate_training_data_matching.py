import argparse
import os
import struct
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad
from scipy.ndimage import binary_dilation

from dfa.audio import Audio
from dfa.utils import read_config


class hp:
    vad_window_length = 30  # In milliseconds
    vad_moving_average_width = 8
    min_voice_length = 4
    vad_sample_rate = 48000


def trim_end(wav):
    int16_max = (2 ** 15) - 1
    samples_per_window = (hp.vad_window_length * hp.vad_sample_rate) // 1000
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=hp.vad_sample_rate))
    voice_flags = np.array(voice_flags)
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    audio_mask = moving_average(voice_flags, hp.vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    audio_mask = np.invert(audio_mask)
    audio_mask[:] = binary_dilation(audio_mask[:], np.ones(hp.min_voice_length + 1))
    audio_mask = np.invert(audio_mask)
    voice_indices = np.where(audio_mask)[0]
    voice_start, voice_end = voice_indices[0], voice_indices[-1]
    audio_mask[:voice_end+hp.min_voice_length] = 1
    audio_mask = np.repeat(audio_mask, samples_per_window)
    return wav[audio_mask]


def get_end(wav):
    int16_max = (2 ** 15) - 1
    samples_per_window = (hp.vad_window_length * hp.vad_sample_rate) // 1000
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=hp.vad_sample_rate))
    voice_flags = np.array(voice_flags)
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    audio_mask = moving_average(voice_flags, hp.vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    audio_mask = np.invert(audio_mask)
    audio_mask[:] = binary_dilation(audio_mask[:], np.ones(hp.min_voice_length + 1))
    audio_mask = np.invert(audio_mask)
    audio_mask = np.repeat(audio_mask, samples_per_window)
    voice_indices = np.where(audio_mask)[0]
    voice_start, voice_end = voice_indices[0], voice_indices[-1]
    return voice_end

class Preprocessor:

    def __init__(self, audio, wav_main, wav_long_main, out_path):
        self.audio = audio
        self.wav_main = wav_main
        self.wav_long_main = wav_long_main
        self.out_path = out_path

    def __call__(self, file_name):
        scores = []
        try:
            print(f'Generating for dir {file_name}')
            wav_path = self.wav_main / file_name
            wav_long_path = self.wav_long_main / f'{file_name[:7]}.wav'

            wavs = sorted(list(wav_path.glob('**/*.wav')))
            wav_long = self.audio.load_wav(wav_long_path)
            starts = []
            start = 0

            for i, wav in enumerate(wavs):
                wav_snippet = self.audio.load_wav(wav)
                window = min(len(wav_snippet), 50000)
                stride = 100
                wav_part = wav_snippet[0:window:stride]
                min_diff = 9999999
                min_t = start
                for t in range(start, min(start + 2500000, len(wav_long) - window)):
                    diff = np.sum(np.abs(wav_long[t:t + window:stride] - wav_part))
                    if diff < min_diff:
                        min_diff = diff
                        min_t = t
                    if min_diff < 15 and diff > min_diff + 10:
                        break
                print(f'{file_name} (start) {wav.stem} {min_diff} {min_t}')
                starts.append((min_t, min_diff))
                scores.append((wav.stem, min_diff))
                start = min_t

            end_startpoints = [len(wav_long)] + list(reversed([s for s, _ in starts])) + [0]

            ends = []
            scores_ends = []
            for i, wav in enumerate(reversed(wavs)):
                wav_snippet = self.audio.load_wav(wav)
                snippet_end = get_end(wav_snippet)
                window = min(len(wav_snippet)-snippet_end, 50000)
                stride = 100
                wav_part = wav_snippet[snippet_end-window:snippet_end:stride]
                min_diff = 9999999
                min_t = end_startpoints[i]
                for t in range(end_startpoints[i], max(end_startpoints[i+1], window), -1):
                    diff = np.sum(np.abs(wav_long[t - window:t:stride] - wav_part))
                    if diff < min_diff:
                        min_diff = diff
                        min_t = t
                print(f'{file_name} (end) {wav.stem} {min_diff} {min_t}')
                min_t += (hp.vad_window_length * hp.vad_sample_rate) // 1000 * (hp.min_voice_length + 1)
                ends.append((min_t, min_diff))
                scores_ends.append((wav.stem, min_diff))

            scores_ends.reverse()
            ends.reverse()

            for i, wav in enumerate(wavs):
                wav_snippet = self.audio.load_wav(wav)
                name = wav.stem
                start = starts[i][0]
                end = ends[i][0]
                wav_cut = wav_long[start:end]
                wav_cut = trim_end(wav_cut)
                print(f'{i} snippet len: {len(wav_snippet)} end-start: {end-start}')
                sf.write(self.out_path / f'{name}.wav', wav_cut, samplerate=self.audio.sample_rate)

            scores = [(a[0], a[1] + b[1]) for a, b in zip(scores, scores_ends)]
            return scores
        except Exception as e:
            print(e)
            return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    args = parser.parse_args()
    config = read_config(args.config)
    audio = Audio.from_config(config['audio'])

    wav_main = Path('/data/datasets/ASVoice4')
    wav_long_main = Path('/data/datasets/ASVoice4_breathing')
    out_path = Path('/data/datasets/ASVoice4_breathing_cutted_2')
    out_path.mkdir(parents=True, exist_ok=True)

    snippet_dirs = sorted(list(next(os.walk(wav_main))[1]))

    preprocessor = Preprocessor(audio=audio, wav_main=wav_main,
                                wav_long_main=wav_long_main, out_path=out_path)

    pool = Pool(processes=1)
    mapper = pool.imap_unordered(preprocessor, snippet_dirs)
    all_scores = []

    for i, scores in enumerate(mapper):
        all_scores.extend(scores)
        all_scores.sort(key=lambda x: -x[1])
        with open(out_path / 'all_scores.txt', 'w', encoding='utf-8') as f:
            lines = [f'{x[0]} {x[1]}\n' for x in all_scores]
            f.writelines(lines)