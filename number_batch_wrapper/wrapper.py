"""ConceptNet Number Batch class."""
import re
from functools import lru_cache
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
            clean_f: Callable[..., str] = clean,
            level: int = 3,
            parallel: bool = True,
    ) -> None:
        """
        Initialise the NumberBatch wrapper.
        
        :param language: Language to encode
        :param path: Path to where NumberBatch files are stored
        :param en_fallback: Whether or not to fallback on the English encodings (requires en-initialisation to work)
        :param clean_f: Cleaning function used before lookup
        :param level: File-depth
        :param parallel: Encode in parallel or not
        """
        self.lang = language
        self.path = path
        self.en_fallback = en_fallback
        self.clean_f = clean_f
        self.parallel = parallel
        self._level = level  # Don't change after init
        self._initialised = self._is_initialised()
    
    def __str__(self) -> str:
        """Representation of the ConceptNetNumberBatch class."""
        return f"NumberBatchWrapper(lang={self.lang})"
    
    def __repr__(self) -> str:
        """Representation of the ConceptNetNumberBatch class."""
        return str(self)
    
    @lru_cache(maxsize=1024)
    def __call__(self, sentences: List[str]) -> np.ndarray:
        """Embed the provided word."""
        if not self._initialised:
            raise NotInitialisedException("Wrapper hasn't been initialised before, please run initialise() first")
        
        # TODO: Continue
        #
        # if lang is not None:
        #     lang = lang.lower()
        #     assert lang in self._LANG
        #
        # # Pre-process the word
        # if len(word.split()) != 1:
        #     raise Exception(f"Only a single word should be given, not '{word}'")
        # word = fold(word.lower())
        #
        # vector = self.get_vector(word, lang=lang)
        # while vector is None:
        #     word = word[:-1]
        #     vector = self.get_vector(word, lang=lang)
        # return vector
    
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
        pbar = tqdm(total=9161912, desc=f"Extracting {lang}..")  # 9161912 hardcoded to improver reading-speed
        try:
            with open(raw_path, 'r') as f:
                line = f.readline()
                while line:
                    pbar.update()
                    if f'/c/{lang}/' == line[:6]:
                        k, v = split_clean(line)
                        if k:
                            data[k] = v
                    line = f.readline()
        finally:
            pbar.close()
            
        # Segment the data
        # TODO
        #  - Perform with self._level: name[:self._level] == file_name
        print(len(data))
    
    def _is_initialised(self) -> bool:
        """Check if the wrapper has been initialised before."""
        return False  # TODO: Check
    
    def get_vector(self, word: str, lang: Optional[str] = None) -> Optional[np.ndarray]:
        """Get the vector of the word stored in the list found under the given path."""
        key = word[:2] if len(word) >= 2 else word[0] * 2
        
        # Try to fetch the word embedding in the original language first
        vector = search_word_emb(word, self._path[lang] / f'{key}.txt') if lang is not None else None
        
        # If nothing found, check other languages (since may be a name)
        if vector is None and lang != 'en':
            vector = search_word_emb(word, self._path['en'] / f'{key}.txt')
        if vector is None and lang != 'fr':
            vector = search_word_emb(word, self._path['fr'] / f'{key}.txt')
        if vector is None and lang != 'nl':
            vector = search_word_emb(word, self._path['nl'] / f'{key}.txt')
        return vector
    
    def add_vector(self, word: str, vector: np.ndarray, languages: Optional[Tuple[str]] = None) -> None:
        """Add a word with corresponding vector to the specified languages."""
        languages = languages if languages else self._LANG
        key = word[:2] if len(word) >= 2 else word[0] * 2
        for lang in languages:
            add_word_emb(word=word, emb=vector, path=self._path[lang] / f'{key}.txt')
    
    def print_overview(self) -> None:
        """Print overview of the class."""
        pass  # TODO


# def search_word_emb(word: str, path: Path) -> Optional[np.ndarray]:
#     """Search for the word embedding in the file specified by the path."""
#     try:
#         with open(str(path), 'r') as f:
#             line = f.readline()
#             while line:
#                 split = line[:-1].split()
#                 if word == split[0]:
#                     return np.asarray(split[1:], dtype=float)
#                 line = f.readline()
#     except FileNotFoundError:
#         raise ConceptNetException(f"Unable to embed '{word}'")
#     return None
#
#
# def add_word_emb(word: str, emb: np.ndarray, path: Path) -> None:
#     """Add a new embedding to the data."""
#     newline = ' '.join([word, ] + [str(e) for e in emb]) + '\n'
#     with open(str(path), 'r') as f:
#         lines = f.readlines()
#
#     # Replace existing if word already in corpus
#     added = False
#     for i, line in enumerate(lines):
#         if line.split()[0] == word:
#             lines[i] = newline
#             added = True
#             break
#
#     # Add new line if not yet in corpus
#     if not added:
#         lines.append(newline)
#
#     with open(str(path), 'w') as f:
#         f.write(''.join(sorted(lines)))
