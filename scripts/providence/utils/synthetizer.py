# Adapted from: gitlab.cognitive-ml.fr/htiteux/paraphone
import asyncio
import itertools
import logging
import random
import shutil
from asyncio import Semaphore
from itertools import zip_longest
from logging import StreamHandler, Formatter
from pathlib import Path
from typing import Optional, Iterable, List, Tuple, Awaitable, Set

import pandas as pd
from aiolimiter import AsyncLimiter
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import texttospeech
from google.cloud.texttospeech_v1 import SynthesisInput
from tqdm.asyncio import tqdm as async_tqdm

VOICES = [
        "en-US-Wavenet-I",    # M
        ]

logger = logging.getLogger("spoken_syntax")
logger.setLevel(logging.DEBUG)
stream_formatter = Formatter("[%(levelname)s] %(message)s")
stream_handler = StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)


class GoogleSpeakSynthesizer:
    STANDARD_VOICE_PRICE_PER_CHAR = 0.000004
    WAVENET_VOICE_PRICE_PER_CHAR = 0.000016
    NUMBER_RETRIES = 4
    RETRY_WAIT_TIME = 10.0

    def __init__(self, lang, voice_id: str, credentials_path: Path):
        self.lang = lang
        self.credentials_file = credentials_path
        self.voice_id = voice_id
        self.voice = texttospeech.VoiceSelectionParams(
            language_code=self.lang,
            name=self.voice_id,
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.OGG_OPUS
        )
        self.client = texttospeech.TextToSpeechAsyncClient.from_service_account_file(str(credentials_path))

    def estimate_price(self, sentences: Iterable[str]):
        return sum(len(sentence) for sentence in sentences) * self.WAVENET_VOICE_PRICE_PER_CHAR

    async def _synth_worker(self, synth_input: SynthesisInput) -> Optional[bytes]:
        for _ in range(self.NUMBER_RETRIES):
            try:
                response = await self.client.synthesize_speech(
                    input=synth_input,
                    voice=self.voice,
                    audio_config=self.audio_config
                )
            except GoogleAPICallError:
                wait_time = random.random() * self.RETRY_WAIT_TIME
                logger.debug(f"Error in synth, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
                continue
            else:
                return response.audio_content
        else:
            return None

    async def synth_text(self, in_tuple: str) -> bytes:
        # in_tuple expected to be (out_filename,text)
        response = await self._synth_worker(texttospeech.SynthesisInput(text=in_tuple[1]))
        return response, in_tuple


class BaseSpeechSynthesisTask:
    MAX_REQUEST_PER_MINUTE = 500
    MAX_REQUEST_PER_SECOND = 12
    MAX_CONCURRENT_REQUEST = 10

    def __init__(self):
        super().__init__()
        self.rate_limiter = AsyncLimiter(self.MAX_REQUEST_PER_SECOND, time_period=1)
        self.semaphore = Semaphore(self.MAX_CONCURRENT_REQUEST)

    def store_output(self, audio_bytes: bytes, sentence: str, folder: Path):
        raise NotImplemented()

    async def tasks_limiter(self, task: Awaitable[Tuple[bytes, List[str]]]):
        async with self.rate_limiter:
            async with self.semaphore:
                return await task

    async def run_synth(self, sentences: List[str], synthesizer: GoogleSpeakSynthesizer, output_folder: Path,
                        test_mode: bool = False):
        synth_tasks = [self.tasks_limiter(synthesizer.synth_text(sentence))
                       for sentence in sentences]
        if test_mode:
            synth_tasks = synth_tasks[:5]

        for synth_task in async_tqdm.as_completed(synth_tasks):
            audio_bytes, sentence = await synth_task
            if audio_bytes is None:
                logger.warning(f"Got none bytes for {sentence}, skipping")
                raise RuntimeError()
            self.store_output(audio_bytes, sentence, output_folder)


class BaseCorporaSynthesisTask(BaseSpeechSynthesisTask):
    SYNTH_SUBFOLDER: str

    def __init__(self, no_confirmation: bool = False):
        super().__init__()
        self.no_confirmation = no_confirmation

    def store_output(self, audio_bytes: bytes, sentence: str, folder: Path):
        out_audio = folder / Path(self.get_filename(sentence))
        out_audio.parent.mkdir(parents=True, exist_ok=True)
        with open(out_audio, "wb") as file:
            file.write(audio_bytes)

    def init_synthesizers(self, credentials_path) -> List[GoogleSpeakSynthesizer]:
        lang = "en-US"
        voices = VOICES
        logger.info(f"Using voices {', '.join(voices)} for synthesis.")
        return [GoogleSpeakSynthesizer(lang, voice_id, credentials_path)
                for voice_id in voices]

    def chunkify(self, iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        for group in zip_longest(*args, fillvalue=fillvalue):
            yield [e for e in group if e is not None]

    @staticmethod
    def get_filename(sentence: Path):
        return sentence[0]

    def get_sentences(self, input_path: Path, test_mode: bool = False) -> Set[str]:
        txt_files = list(input_path.glob('**/*.txt'))
        if test_mode:
            txt_files = txt_files[:4]

        out = []
        for txt_file in txt_files:
            with open(txt_file, 'r') as fin:
                sentence = fin.read()
            out.append((txt_file.relative_to(input_path).with_suffix('.ogg'), sentence[:-1]))
        return out

    def run(self, input_path, output_folder, credentials_path, test_mode):
        synth_folder = output_folder

        logger.info("Parsing sentences...")
        all_sentences = self.get_sentences(input_path, test_mode)
        logger.info(f"Found {len(all_sentences)} sentences to synthesize")

        synthesizers = self.init_synthesizers(credentials_path)
        synth_sentences = {
            synth: list(all_sentences) for synth in synthesizers
        }
        # filtering sentences that might already have been generated
        for synth, sentences in list(synth_sentences.items()):

            audio_folder = synth_folder
            if not audio_folder.exists():
                continue
            elif not list(audio_folder.iterdir()):
                continue
            synth_sentences[synth] = [
                sentence for sentence in sentences
                if not (audio_folder / Path(synth.voice_id) / Path(self.get_filename(sentence))).exists()
            ]
            logger.info(f"{len(sentences) - len(synth_sentences[synth])} sentences "
                        f"already exist for {synth.voice_id} and won't be synthesized")

        total_cost = sum(synth.estimate_price(sentences)
                         for synth, sentences in synth_sentences.items())
        logger.info(f"Estimated cost is {total_cost}$")
        if not self.no_confirmation:
            if input("Do you want to proceed?\n[Y/n]").lower() != "y":
                logger.info("Aborting")
        #         return

        logger.info("Starting synthesis...")
        loop = asyncio.get_event_loop()
        for synth, words in synth_sentences.items():
            for words_chunk in self.chunkify(words, 2 ** 12):
                logger.info(f"For synth with voice id {synth.voice_id}")
                audio_folder = synth_folder / Path(synth.voice_id)
                audio_folder.mkdir(parents=True, exist_ok=True)

                async_tasks = self.run_synth(words_chunk,
                                             synth,
                                             audio_folder)
                loop.run_until_complete(async_tasks)