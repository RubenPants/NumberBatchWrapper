"""Utilisation methods."""
import gzip
import re
import shutil
from pathlib import Path
from typing import Any, Optional, Set
from urllib.request import urlretrieve

from fold_to_ascii import fold
from tqdm import tqdm


def clean(x: str) -> str:
    """Default cleaning method, put text to lowercase, remove non-letters and folds to ASCII."""
    x = fold(x.lower())
    x = re.sub(r'[^a-z]', '', x)
    return x


def get_raw_nb_path(
        path: Path,
        version: str = '19.08',
) -> Path:
    """
    Load in the raw multilingual NumberBatch data.
    
    :param path: Folder where data is stored (or should be stored once downloaded)
    :param version: Version of Multilingual NumberBatch data to download, if not yet exists
    :return: Path pointing to unzipped Multilingual NumberBatch file
    """
    assert path.is_dir()
    download(path, version=version)  # Download file if not yet exists
    return path / f"numberbatch-{version}.txt"


def tqdm_hook(t: tqdm) -> Any:
    """Progressbar to visualisation downloading progress."""
    last_b = [0]
    
    def update_to(b: int = 1, bsize: int = 1, t_size: Optional[int] = None) -> None:
        if t_size is not None:
            t.total = t_size
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    
    return update_to


def download(
        destination: Path,
        version: str = '19.08',
) -> None:
    """
    Download the multilingual NumberBatch data, as specified by the given version.

    :param destination: Destination path where downloaded result is stored
    :param version: Version of Multilingual NumberBatch data to download
    :return: Success of the download
    """
    # If already downloaded, return
    if (destination / f"numberbatch-{version}.txt").is_file():
        return
    
    # File does not yet exist, download it (can take a while)
    if not (destination / f"numberbatch-{version}.txt.gz").is_file():
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading..") as pbar:
            urlretrieve(
                    f"https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-{version}.txt.gz",
                    destination / f"numberbatch-{version}.txt.gz",
                    reporthook=tqdm_hook(pbar),
            )
    
    # Unzip the downloaded file to a txt document
    with tqdm(desc="Unzipping"):
        with gzip.open(destination / f"numberbatch-{version}.txt.gz", 'rb') as f_in:
            with open(destination / f"numberbatch-{version}.txt", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def get_special_tokens() -> Set[str]:
    """Get a set of regularly occurring special tokens."""
    return {
        'ϳ', 'ȑ', 'б', '‘', '書', 'ה', 'ঁ', 'べ', 'ﻤ', 'ԛ', 'あ', '🂾', 'ǁ', '見', 'μ', '子', '☲', 'ƿ', 'ﻋ', 'せ', 'ș', '㍳',
        'ﺩ', '⚊', 'ク', '段', 'ȁ', '頭', 'お', '己', '™', 'ョ', 'ź', '櫛', '目', 'ピ', 'ﺀ', 'ň', '画', '🂻', '프', '小', '絵', '冗',
        'ん', 'ꞌ', 'ợ', 'ł', 'ﻓ', 'ל', 'ﺘ', '券', 'č', 'ā', 'ɩ', 'ő', 'ḷ', '髪', '♪', '機', '℧', '半', 'む', 'ạ', 'ﻌ', 'ﺠ',
        '©', 'ﺬ', '🃛', 'え', 'ν', 'ﺽ', 'י', '飯', 'ن', 'ø', 'ǘ', 'り', '̚', 'ʰ', '砲', 'α', '聞', '足', 'ả', '鳥', '⚍', '♥',
        'ℝ', 'ț', 'な', 'ザ', 'に', 'ɲ', '席', '✔', '̐', 'ぬ', 'ʔ', '階', '心', '賊', 'ζ', '紙', 'ʹ', '♂', '●', 'ϻ', 'ﻭ', 'ˑ',
        'ﺼ', 'ȕ', 'っ', 'ﻃ', '慈', '↗', 'ィ', '🃎', 'ụ', 'а', '甲', 'ằ', 'ば', '𝄢', 'ﻨ', 'ة', 'ﺔ', 'å', 'ガ', '波', '花', '議',
        '̇', 'χ', '讚', 'ﻡ', 'ө', 'ɣ', 'ű', 'サ', '↘', '⠴', '撃', '✓', 'ở', '法', 'ﻐ', '№', 'ﻀ', '馬', 'φ', '·', 'ف', 'も',
        'ġ', 'ϛ', 'א', '泣', '泉', '草', '⠧', 'ﻮ', 'ṟ', '⠲', '한', '仕', '境', 'þ', '家', '☆', '粧', 'ﻝ', '☰', '戊', '気', 'と',
        '隠', '酒', '肉', 'к', 'и', 'ו', 'ﻘ', 'じ', 'は', 'ﺞ', '⚌', 'е', '場', '☯', 'ﺱ', 'ﺨ', 'バ', 'ℤ', 'ﻎ', '♩', 'ξ', '損',
        '藤', 'ǔ', 'ץ', 'ҫ', '糸', '卯', '酉', '櫂', 'ﻥ', '⠚', '敵', 'ז', '丑', 'ğ', 'ŵ', '𝄞', '象', 'ᵐ', '会', 'ﺵ', 'ᵣ', '랑',
        'अ', '煙', 'う', '駕', 'م', 'ẩ', 'ل', '’', 'ž', 'θ', 'ȍ', '程', 'ꭓ', 'ο', 'ﺷ', 'º', 'ª', '下', 'イ', 'ʾ', '房', 'ャ',
        'ﺑ', '有', 'パ', 'し', 'ɵ', '午', 'ご', 'デ', '☳', '塵', '容', 'ן', 'ộ', 'ア', '生', '威', '船', '都', 'ʊ', 'ś', 'ر', 'ﺯ',
        '霧', 'ė', 'η', 'ỉ', '学', 'ﺟ', '°', '食', 'ך', 'ş', 'š', 'ừ', '̓', 'よ', '衛', '権', 'ﺐ', 'へ', '⚎', '神', '麦', '़',
        'ế', '魚', 'м', '℀', '演', '火', 'ﺲ', 'ﻚ', 'ロ', '潜', 'ﻠ', 'き', 'ɨ', '衣', '스', 'ū', '死', '薪', 'ズ', 'đ', 'ﺴ', 'コ',
        '☥', 'ﻕ', '問', 'ﹻ', 'ё', '料', '℟', 'る', 'ﻏ', 'こ', '胃', 'ﻙ', '☷', 'ƶ', '⠉', 'れ', 'ド', '🃝', 'ゅ', 'ч', 'ʃ', '山',
        'ﻦ', '空', 'צ', '業', 'ь', 'ﺰ', '☉', 'ƴ', '光', 'ﺮ', 'ఁ', 'ọ', 'プ', 'ṭ', '理', 'כ', 'タ', '⠐', 'ŝ', '̀', '安', 'ひ',
        'ı', 'く', 'ى', '獣', '巳', '─', 'ﻁ', '͂', 'ש', 'و', 'í', 'ﺖ', 'ƒ', '炉', 'ﺿ', '鋤', '音', 'ン', '談', 'ъ', '岐', '戌',
        '快', 'ﻍ', '🃋', 'ǎ', 'λ', 'κ', 'ﻖ', '汽', 'ể', '⠈', 'ラ', 'ボ', 'ﺕ', 'ﺳ', '🚃', 'ń', '語', '連', '舞', '籠', '어', '話',
        '̆', '形', 'ﺺ', '을', 'ﺹ', 'ﻣ', '艦', 'ự', '🃍', '́', 'ט', 'ĩ', 'ﺸ', '債', 'ビ', 'ﺒ', 'ε', 'テ', 'ʋ', 'ţ', '字', '寅',
        'л', 'オ', '\u200d', 'ꞵ', '⠎', '®', '辛', 'ᱚ', '☱', 'ס', 'ē', '化', '국', '触', '海', 'œ', '⠒', 'け', 'ま', '声', 'ﺧ',
        '葬', 'ʒ', '✡', 'ɂ', '⠨', '⚏', 'ר', '丁', '指', '東', 'ﻜ', '無', 'ơ', '̄', 'ي', '♠', 'о', 'ż', '壬', 'ψ', '噌', '鶴',
        '劇', '営', '風', 'ע', '列', '名', 'ʳ', '鸛', 'ŏ', '女', '台', 'レ', 'ب', '⠤', '寸', 'ắ', 'з', 'ג', 'ワ', '四', 'ː', 'ଁ',
        'ﻟ', '庚', '⠏', 'ﺁ', 'מ', 'ﺶ', '関', 'ઁ', 'γ', '式', 'ᱢ', '☵', '勢', '計', 'र', 'ǃ', '服', 'ﺓ', 'ư', '̍', 'ﺥ', '絡',
        'τ', 'ậ', 'キ', 'ǂ', 'נ', 'ت', 'ח', 'ﻪ', '☴', '満', '祝', 'ー', 'や', 'ँ', 'ữ', 'ą', '霰', '夫', 'い', 'ら', 'ﻗ', 'ﺏ',
        '矢', 'ﻧ', '谷', 'ﺗ', '〆', '℣', '🃞', '浅', 'õ', 'ғ', 'の', 'シ', '☃', 'ノ', 'ग', 'ﺦ', 'х', '葉', 'д', '伊', '田', 'ᵉ',
        '丙', 'ə', '亥', '肴', 'グ', '水', '握', '尊', '申', '癸', 'ﺭ', '香', 'で', '者', '林', '物', 'ﻞ', 'ا', 'פ', '屏', '敬', 'ゃ',
        'ρ', 'ﻄ', '詞', '⠗', '両', 'ℵ', 'ầ', 'ȅ', '年', 'ﺪ', '妻', '屋', '♦', '魔', 'ﺝ', '्', 'て', '♀', 'с', 'ŭ', '大', '江',
        'δ', 'ǫ', 'ᱡ', 'ペ', '禁', 'ろ', 'ﺾ', 'す', 'ホ', '⠇', '未', '今', 'ਁ', 'ᱟ', '乙', 'ă', 'ồ', '度', '痛', '顔', '♫', 'ñ',
        'у', '辰', '♮', '櫓', 'ル', 'ﻩ', 'ם', '宿', 'ʻ', '‿', 'ǐ', '☶', '姿', '⌀', '衿', 'を', 'я', '傷', 'ę', 'ℂ', 'ш', '̏',
        '🂫', 'ɚ', 'ﻬ', 'だ', '車', '月', 'ý', '🂭', '惑', '🂽', '글', '牛', '確', '職', '熱', 'ó', 'ю', '狼', 'ι', '〒', 'ŋ', '帯',
        'ぼ', 'ת', '鶏', '戦', '可', 'σ', '烏', '束', 'リ', '体', '国', 'ﻯ', '狐', 'н', '電', '⚋', 'ɓ', 'ﻢ', 'た', '楽', 'т', '⚷',
        '前', '♣', 'ב', 'ス', 'ず', 'ﻛ', 'ị', '星', '⠛', 'ř', 'ﻫ', 'ﻊ', '𝄡', 'ấ', 'ु', 'ω', '映', '♭', 'ō', 'ᱠ', '\u200e',
        'ハ', '檎', 'ヘ', 'ﺫ', 'ờ', 'π', 'ﻰ', 'ì', 'ḳ', 'が', '℞', '◌', 'ệ', '🂮', 'β', 'ã', '秘', '具', 'ȉ', 'ﻂ', 'ד', '味',
        'ī', '拳', '後', '鯨', 'ⓚ', 'ﻒ', 'ף', 'ð', '合', 'ء', 'υ', 'ﻔ', 'ỏ', '文', 'つ', '⠙', '毒', 'ѳ', 'か', 'р', 'み', '事',
        'ɛ', 'ع', 'ﻑ', 'в', 'ү', 'ɔ', '割', 'ﺻ', '卐', 'ũ', 'ɜ', '̂', 'ʿ', 'ṣ', 'ề',
    }


def get_numbers() -> Set[str]:
    """Get a set of numbers (as strings)."""
    return {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}


def get_symbols() -> Set[str]:
    """Get a set of symbols."""
    return {' ', '"', '#', "'", '.', '_', 'ǀ'}
