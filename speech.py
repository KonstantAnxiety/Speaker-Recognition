import json
import os
import tempfile
import wave
from collections import namedtuple
from typing import List

import soundfile as sf
import matplotlib.pyplot as plt
from vosk import Model, KaldiRecognizer
from simple_diarizer.diarizer import Diarizer
from simple_diarizer.utils import combined_waveplot


class Word:
    """ A class representing a word from the JSON format for vosk speech recognition API """

    def __init__(self, input_dict):
        """
        Parameters:
          input_dict (dict) dictionary from JSON, containing:
            conf (float): degree of confidence, from 0 to 1
            end (float): end time of the pronouncing the word, in seconds
            start (float): start time of the pronouncing the word, in seconds
            word (str): recognized word
        """

        self.conf = input_dict["conf"]
        self.end = input_dict["end"]
        self.start = input_dict["start"]
        self.word = input_dict["word"]

    def to_string(self):
        """ Returns a string describing this instance """
        return "{:20} from {:.2f} sec to {:.2f} sec, confidence is {:.2f}%".format(
            self.word, self.start, self.end, self.conf*100)


def sec_to_frames(sec_start: float, sec_end: float, sample_rate: int = 48000) -> int:
    return int((sec_end - sec_start) * sample_rate)


class SpeechRecognizer:
    def __init__(self, input_path: str, num_speakers: int, ffmpeg_path: str):
        self.input_path = input_path
        self.num_speakers = num_speakers
        self.ffmpeg_path = ffmpeg_path
        self.diar_segments = None
        self.list_of_words = None

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.tmp_dir = os.path.join(dir_path, 'tmp_files')
        self.vosk_ru_model_path = os.path.join(dir_path, 'models', 'vosk-model-ru-0.22')
        # inputfile = os.path.join(dir_path, 'input', 'gentlemen.mp3')
        self.wavfile = self.input_path + '.wav'

    def convert_to_wav(self):
        ffmpeg = os.path.join(self.ffmpeg_path, 'ffmpeg-2022-03-07', 'bin', 'ffmpeg.exe')  # r'/usr/local/bin/ffmpeg'
        ffmpeg_pipe = ' '.join((ffmpeg, '-y', '-i', self.input_path, '-ar', '48000', '-ac', '1', '-f', 'wav', self.wavfile))
        if not os.path.exists(self.wavfile):
            print('Converting...')
            os.system(ffmpeg_pipe)
            print('Done converting.')

    def diarize(self):
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as outdir:
            signal, fs = sf.read(self.wavfile)

            diar = Diarizer(
                embed_model='ecapa',  # supported types: ['xvec', 'ecapa']
                cluster_method='sc',  # supported types: ['ahc', 'sc']
                window=1.5,  # size of window to extract embeddings (in seconds)
                period=0.75  # hop of window (in seconds)
            )

            if self.num_speakers > 0:
                self.diar_segments = diar.diarize(
                    self.wavfile,
                    num_speakers=self.num_speakers,
                    outfile=f"{outdir}/result.rttm"
                )
            else:
                self.diar_segments = diar.diarize(
                    self.wavfile,
                    num_speakers=None,
                    threshold=1e-1,
                    outfile=f"{outdir}/result.rttm"
                )
        if 0 < self.num_speakers < 10:
            combined_waveplot(signal, fs, self.diar_segments, figsize=(10, 3), tick_interval=60)
            plt.show()

    def speech_to_text(self):
        wf = wave.open(self.wavfile, "rb")

        vosk_model = Model(self.vosk_ru_model_path)
        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        rec.SetWords(True)
        results = []

        # wf.rewind()
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part_result = json.loads(rec.Result())
                results.append(part_result)
        part_result = json.loads(rec.FinalResult())
        results.append(part_result)

        # convert list of JSON dictionaries to list of 'Word' objects
        list_of_words: List[Word] = []
        for sentence in results:
            if len(sentence) == 1:
                continue
            for obj in sentence['result']:
                w = Word(obj)
                list_of_words.append(w)
        wf.close()
        self.list_of_words = list_of_words

    def finilize(self) -> str:
        DialogueLine = namedtuple('DialogueLine', ('speaker', 'text'))
        dialogue: List[DialogueLine] = []
        eps = 0.7

        current_word_i = 0
        for segment in self.diar_segments:
            line = ''
            while current_word_i < len(self.list_of_words) and segment['start'] - eps >= self.list_of_words[current_word_i].start:
                current_word_i += 1
            while current_word_i < len(self.list_of_words) and segment['start'] - eps <= (
                    self.list_of_words[current_word_i].start + self.list_of_words[current_word_i].start) / 2 <= segment[
                'end'] + eps:
                line += self.list_of_words[current_word_i].word + ' '
                current_word_i += 1
            if line != '':
                dialogue.append(DialogueLine(speaker=segment['label'], text=line))

        res_text = ''
        for line in dialogue:
            # dist = cosine_dist(line.spk_sig, avg_spk_sig)
            res_text += f'Speaker #{line.speaker}: {line.text}\n'
            print(f'Speaker #{line.speaker}: {line.text}')

        return res_text
