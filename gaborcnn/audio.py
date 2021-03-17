import os
from typing import Optional

from librosa import logamplitude
from librosa.feature import melspectrogram, mfcc
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment


class Clip:
    RATE = 44100.
    FRAME = 512

    class Audio:
        def __init__(self, path: str, silence: int = 5000) -> None:
            self.path = path
            self.silence = silence
            self.data: Optional[AudioSegment] = None
            self.raw: Optional[np.ndarray] = None

        def __enter__(self):
            self.data = AudioSegment.silent(duration=self.silence).overlay(
                AudioSegment.from_file(self.path)[5:self.silence])
            self.raw = ((np.fromstring(self.data._data, dtype='int16') + 0.5)
                        / (0x7FFF + 0.5))
            return self

        def __exit__(self, exc_type, exc, exc_tb):
            if exc_type:
                print(f'{exc_type}\n{exc}\n{exc_tb}')
            del self.data
            del self.raw

    def __init__(self, filename: str) -> None:
        self.filename = os.path.basename(filename)
        self.path = os.path.abspath(filename)
        self.directory = os.path.dirname()

        self.category = self.directory.split('/')[-1]
        self.audio = Clip.Audio(self.path)

        with self.audio as audio:
            self._compute_mfcc(audio)
            self._compute_scr(audio)

    def __repr__(self) -> str:
        return f'<{self.category}/{self.filename}>'

    def _compute_mfcc(self, audio) -> None:
        self.melspec = melspectrogram(audio.raw, sr=Clip.RATE,
            hop_length=Clip.FRAME)
        self.logamp = logamplitude(self.melspec)
        self.mfcc = mfcc(S=self.logamp, n_mfcc=13).transpose()

    def _compute_scr(self, audio) -> None:
        frames = int(np.ceil(len(audio.data) / 1000.
                    * Clip.RATE / Clip.FRAME))
        self.zcr = np.asarray([np.mean(.5 * np.abs(np.diff(np.sign(
            Clip._get_frame(audio, i))))) for i in range(frames)])

    def display_audio(self):
        fig, ax0 = plt.subplot(2, 1, 1)
        ax0.set_title(f'{self.category} : {self.filename}')
        ax0.plot(np.arange(0, len(self.audio.raw)) / Clip.RATE, self.audio.raw)


    @staticmethod
    def _get_frame(audio, index):
        return (None if index < 0 else
            audio.raw[(index * Clip.FRAME):(index+1) * Clip.FRAME])

