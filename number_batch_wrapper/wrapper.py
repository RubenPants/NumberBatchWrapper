"""ConceptNet Number Batch class."""
import re
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from number_batch_wrapper.utils import clean, get_numbers, get_raw_nb_path, get_special_tokens, get_symbols


class EncodingException(Exception):
    """Exception thrown when a failure occurs during encoding."""
    
    pass


class NotInitialisedException(Exception):
    """Exception thrown when NumberBatch class has not been initialised."""
    
    pass


class Wrapper:
    """Wrapper for the multilingual Number Batch data to improve encoding speed."""
    
    SIZE = 300
    
    def __init__(
            self,
            language: str,
            path: Path,
            en_fallback: bool = True,
            normalise: bool = True,
            clean_f: Callable[..., str] = clean,
            tokenizer: Callable[..., str] = lambda x: x.split(),
            level: int = 3,
    ) -> None:
        """
        Initialise the NumberBatch wrapper.
        
        :param language: Language to encode
        :param path: Path to where NumberBatch files are stored
        :param en_fallback: Whether or not to fallback on the English encodings (requires en-initialisation to work)
        :param normalise: Normalise the resulting embeddings
        :param clean_f: Cleaning function used before lookup
        :param tokenizer: Function to split sentence into words (split on whitespace by default)
        :param level: File-depth
        """
        assert level > 0
        self.lang = language
        self.path = path
        self.en_fallback = en_fallback and language != 'en'
        self.normalise = normalise
        self.clean_f = clean_f
        self.tokenizer = tokenizer
        self._level = level  # Don't change after init
    
    def __str__(self) -> str:
        """Representation of the ConceptNetNumberBatch class."""
        return f"NumberBatchWrapper(lang={self.lang})"
    
    def __repr__(self) -> str:
        """Representation of the ConceptNetNumberBatch class."""
        return str(self)
    
    def __call__(self, sentences: List[str]) -> np.ndarray:
        """Embed the provided word."""
        if not self.is_initialised():
            raise NotInitialisedException("Wrapper hasn't been initialised before, please run initialise() first")
        
        result = []
        for sentence in tqdm(sentences, desc="Embedding"):
            result.append(self.embed(sentence))
        return np.vstack(result)
    
    def embed(self, sentence: str) -> np.ndarray:
        """Embed a single sentence."""
        words = [self.clean_f(w) for w in self.tokenizer(sentence)]
        
        # Embed the sequence of words
        result = np.zeros((self.SIZE,))
        for word in words:
            v = _read_file(
                    word=word,
                    file=self.path / f"_nb_{self.lang}/{word[:self._level]}",
                    alternative_file=self.path / f"_nb_en/{word[:self._level]}" if self.en_fallback else None,
            )
            if v is not None: result += v  # If v is None; ignore
        
        # Normalise the result if requested
        norm = result.sum()
        if not norm: raise EncodingException(f"No encoding found for sentence '{sentence}'")
        if self.normalise: result /= norm
        return result
    
    def initialise(
            self,
            inp_path: Path,
            version: str = "19.08",
            letters_only: bool = True,
            remove_special_tokens: bool = True,
            remove_symbols: bool = True,
            remove_numbers: bool = True,
    ) -> None:
        """
        Initialise the instance, should only be run once per machine.
        
        :param inp_path: Folder where raw NumberBatch data is stored, or will be stored once downloaded
        :param version: NumberBatch version to use
        :param letters_only: Only consider words strictly consisting of letters
        :param remove_special_tokens: Remove words containing special tokens
        :param remove_symbols: Remove words containing symbols
        :param remove_numbers: Remove words containing numbers
        """
        assert inp_path.is_dir()
        
        # Get path to raw multilingual encodings
        raw_path = get_raw_nb_path(inp_path, version=version)
        
        # Initialise
        self._initialise_lang(
                lang=self.lang,
                raw_path=raw_path,
                letters_only=letters_only,
                remove_special_tokens=remove_special_tokens,
                remove_symbols=remove_symbols,
                remove_numbers=remove_numbers,
        )
        if self.en_fallback and self.lang != 'en':
            self._initialise_lang(
                    lang='en',
                    raw_path=raw_path,
                    letters_only=letters_only,
                    remove_special_tokens=remove_special_tokens,
                    remove_symbols=remove_symbols,
                    remove_numbers=remove_numbers,
            )
    
    def _initialise_lang(
            self,
            lang: str,
            raw_path: Path,
            letters_only: bool,
            remove_special_tokens: bool,
            remove_symbols: bool,
            remove_numbers: bool,
    ) -> None:
        """Initialise the multilingual data by language."""
        special_tokens = get_special_tokens() if remove_special_tokens else set()
        symbols = get_symbols() if remove_symbols else set()
        numbers = get_numbers() if remove_numbers else set()
        
        def split_clean(line: str) -> Tuple[Optional[str], List[float]]:
            """Split and clean the given line."""
            name = line.split()[0][6:]
            
            # Filter if invalid name
            if letters_only and re.match(r'[^a-zA-Z]', name):
                return None, []
            name_set = set(name)
            if name_set & special_tokens:
                return None, []
            if name_set & symbols:
                return None, []
            if name_set & numbers:
                return None, []
            
            # Valid name, return cleaned name together with corresponding vector
            return self.clean_f(name), [float(x) for x in line.split()[1:]]
        
        # Extract all data related to this language
        data = {}
        pbar = tqdm(total=9161912, desc=f"Extracting '{lang}'..")  # 9161912 hardcoded to improver reading-speed
        try:
            with open(raw_path, 'r') as f:
                line = f.readline()
                while line:
                    pbar.update()
                    if f'/c/{lang}/' == line[:6]:
                        k, v = split_clean(line)
                        if k: data[k] = v
                    line = f.readline()
        finally:
            pbar.close()
        
        # Segment the data
        with tqdm(desc=f"Segmenting '{lang}'.."):
            files = {}
            for word in data.keys():
                tag = word[:self._level]
                if tag not in files: files[tag] = set()
                files[tag].add(word)
            
            def process(word: str, vector: List[float]) -> str:
                """Turn word and vector couple to string."""
                return f"{word} {' '.join([str(v) for v in vector])}"
            
            path = self.path / f"_nb_{lang}"
            path.mkdir(exist_ok=True, parents=True)
            for tag, words in files.items():
                with open(path / tag, 'w') as f:
                    f.write('\n'.join([process(word, data[word]) for word in words]))
    
    def is_initialised(self) -> bool:
        """Check if the wrapper has been initialised before."""
        if not glob(str(self.path / f"_nb_{self.lang}/*")): return False
        if self.en_fallback and not glob(str(self.path / f"_nb_en/*")): return False
        return True


@lru_cache(maxsize=1024)
def _read_file(
        word: str,
        file: Path,
        alternative_file: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Read in the vector associated with the provided word.
    
    :param word: The word to look for
    :param file: The file to search over for the word
    :param alternative_file: Alternative file (e.g. in other language) to search over for the word
    """
    # Lookup the word
    if file.is_file():
        with open(file, 'r') as f:
            line = f.readline()
            while line:
                split = line.split()
                if word == split[0]:
                    return np.asarray(split[1:], dtype=float)
                line = f.readline()
                
    # Not found in requested language, check if available in English
    if alternative_file and alternative_file.is_file():
        with open(alternative_file, 'r') as f:
            line = f.readline()
            while line:
                split = line.split()
                if word == split[0]:
                    return np.asarray(split[1:], dtype=float)
                line = f.readline()
                
    # If still nothing found, prune the word's tail and try again
    if len(word) > 1:
        return _read_file(
                word=word[:-1],
                file=file,
                alternative_file=alternative_file,
        )
    
    # Still nothing found, return None
    return None
