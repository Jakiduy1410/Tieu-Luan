import re
import warnings
from functools import lru_cache
from typing import Callable

import pandas as pd


VIETNAMESE_STOPWORDS = {
    "a", "ai", "anh", "ay", "ba", "ban", "bao", "bi", "biet", "boi",
    "ca", "cai", "cac", "can", "cang", "chi", "cho", "chu", "chua",
    "chung", "co", "con", "cua", "da", "dang", "day", "de", "den",
    "deu", "di", "do", "du", "duoc", "em", "gi", "giua", "hay", "hon",
    "khi", "khac", "la", "lai", "lam", "len", "luc", "luon", "ma",
    "moi", "mot", "minh", "muon", "nao", "nay", "nen", "neu", "nhu",
    "nhung", "nua", "o", "phai", "qua", "ra", "rang", "rat", "roi",
    "sau", "se", "so", "su", "tai", "theo", "thi", "toi", "tren",
    "trong", "truoc", "tu", "tung", "vao", "ve", "vi", "voi", "vua",
    "va", "van",
    "à", "á", "ạ", "ả", "ã", "ấy", "bà", "bài", "bạn", "bằng", "bị",
    "biết", "bộ", "bởi", "cả", "cái", "các", "cần", "càng", "cao",
    "câu", "chắc", "chỉ", "cho", "chứ", "chưa", "chúng", "chúng_ta",
    "chúng_tôi", "có", "còn", "cùng", "cũng", "của", "cuối", "đã",
    "đang", "đây", "đấy", "để", "đến", "đều", "đi", "đó", "được",
    "dù", "do", "dưới", "em", "gì", "giữa", "gần", "hay", "hơn",
    "hết", "khi", "khác", "khoảng", "là", "lại", "làm", "lên", "lúc",
    "luôn", "mà", "mỗi", "một", "mọi", "mình", "muốn", "nào", "này",
    "nên", "nếu", "ngay", "người", "như", "nhưng", "những", "nơi",
    "nữa", "ở", "phải", "qua", "ra", "rằng", "rất", "rồi", "sau",
    "sẽ", "so", "sự", "tại", "theo", "thì", "trên", "trong", "trước",
    "từ", "từng", "tới", "tôi", "tớ", "và", "vẫn", "vào", "vậy", "về",
    "vì", "việc", "với", "vừa", "xin", "ông", "đâu",
}


def _normalize_input(text) -> str:
    if text is None or pd.isna(text):
        return ""
    return str(text)


def none_preprocess(text) -> str:
    return _normalize_input(text)


def basic_preprocess(text) -> str:
    text = _normalize_input(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=1)
def _get_word_segmenter():
    try:
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize
    except ImportError:
        pass

    try:
        from underthesea import word_tokenize

        return lambda text: word_tokenize(text, format="text")
    except ImportError:
        warnings.warn(
            "pyvi/underthesea is not installed; advanced preprocessing will skip word segmentation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _word_segment(text: str) -> str:
    if not text:
        return ""

    segmenter = _get_word_segmenter()
    if segmenter is None:
        return text
    return segmenter(text)


def _remove_stopwords(text: str) -> str:
    if not text:
        return ""
    return " ".join(token for token in text.split() if token not in VIETNAMESE_STOPWORDS)


def advanced_preprocess(text) -> str:
    text = basic_preprocess(text)
    text = _word_segment(text)
    text = _remove_stopwords(text)
    return re.sub(r"\s+", " ", text).strip()


def get_preprocessor(method: str) -> Callable:
    preprocessors = {
        "none": none_preprocess,
        "basic": basic_preprocess,
        "advanced": advanced_preprocess,
    }
    method = method.lower().strip()
    if method not in preprocessors:
        raise ValueError("method must be one of: none, basic, advanced")
    return preprocessors[method]


def apply_preprocessing(
    df: pd.DataFrame,
    method: str = "none",
    text_column: str = "Comment",
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' was not found.")

    processed = df.copy()
    processed[text_column] = processed[text_column].apply(get_preprocessor(method))

    columns_to_keep = [text_column]
    for label_col in ["Label", "label", "SpamLabel", "spam_label", "Rating", "rating"]:
        if label_col in processed.columns and label_col not in columns_to_keep:
            columns_to_keep.append(label_col)

    return processed[columns_to_keep]
