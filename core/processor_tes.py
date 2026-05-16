"""
processor.py

Module tiền xử lý văn bản cho bài toán phân loại tin nhắn/bình luận Spam vs Ham.

Gồm 3 mức xử lý:
- none: giữ nguyên văn bản, chỉ xử lý NaN/None
- basic: lowercase, xóa dấu câu/ký tự đặc biệt/chữ số, chuẩn hóa khoảng trắng
- advanced: basic + tách từ tiếng Việt + xóa stopwords

Ngoài ra, file này có thể đọc toàn bộ data/test.csv và xuất ra 3 file CSV:
- data/preprocessed/test_none.csv
- data/preprocessed/test_basic.csv
- data/preprocessed/test_advanced.csv
"""

import re
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd


# Danh sách stopwords tiếng Việt cơ bản.
# Có thể bổ sung thêm tùy theo dữ liệu thực tế.
VIETNAMESE_STOPWORDS = {
    "a", "ai", "anh", "ấy", "ạ", "à", "á", "ả", "ã", "bà", "bài", "bạn", "bao",
    "bằng", "bị", "biết", "bộ", "bởi", "cả", "cái", "các", "cần", "càng", "cao",
    "câu", "chắc", "chỉ", "cho", "chứ", "chưa", "chúng", "chúng_ta", "chúng_tôi",
    "có", "còn", "cùng", "cũng", "của", "cuối", "đã", "đang", "đây", "đấy",
    "để", "đến", "đều", "đi", "đó", "được", "dù", "do", "dưới", "em", "gì", "giữa",
    "gần", "hay", "hơn", "hết", "khi", "khác", "khoảng", "là", "lại", "làm", "lên",
    "lúc", "luôn", "mà", "mỗi", "một", "mọi", "mình", "muốn", "nào", "này", "nên",
    "nếu", "ngay", "người", "như", "nhưng", "những", "nơi", "nữa", "ở", "phải",
    "qua", "ra", "rằng", "rất", "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", "thì",
    "trên", "trong", "trước", "từ", "từng", "tới", "tôi", "tớ", "và", "vẫn", "vào",
    "vậy", "về", "vì", "việc", "với", "vừa", "xin", "ông", "đâu",
}


def _normalize_input(text) -> str:
    """
    Chuyển input về chuỗi an toàn.
    NaN/None -> "".
    """
    if text is None or pd.isna(text):
        return ""
    return str(text)


def none_preprocess(text) -> str:
    """
    Giữ nguyên văn bản gốc, chỉ xử lý lỗi NaN/None.
    """
    return _normalize_input(text)


def basic_preprocess(text) -> str:
    """
    Tiền xử lý cơ bản:
    - Chuyển về chữ thường
    - Xóa URL, email
    - Xóa dấu câu, ký tự đặc biệt, emoji
    - Xóa chữ số
    - Chuẩn hóa khoảng trắng
    """
    text = _normalize_input(text).lower()

    # Xóa URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Xóa email
    text = re.sub(r"\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " ", text)

    # Xóa số
    text = re.sub(r"\d+", " ", text)

    # Xóa dấu câu, ký tự đặc biệt, emoji nhưng giữ chữ Unicode tiếng Việt
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Xóa dấu gạch dưới nếu có
    text = re.sub(r"_+", " ", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _word_segment(text: str) -> str:
    """
    Tách từ tiếng Việt.
    Ưu tiên pyvi, nếu không có thì dùng underthesea.
    Nếu cả hai thư viện đều chưa cài, trả về text basic để file vẫn chạy được.
    """
    if not text:
        return ""

    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    except ImportError:
        pass

    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        warnings.warn(
            "Chưa cài pyvi hoặc underthesea nên advanced_preprocess sẽ bỏ qua bước tách từ. "
            "Có thể cài bằng: pip install pyvi hoặc pip install underthesea",
            RuntimeWarning,
            stacklevel=2,
        )
        return text


def _remove_stopwords(text: str) -> str:
    """
    Xóa stopwords khỏi câu đã tách từ.
    """
    if not text:
        return ""

    tokens = text.split()
    filtered_tokens = [
        token for token in tokens
        if token not in VIETNAMESE_STOPWORDS
    ]

    return " ".join(filtered_tokens)


def advanced_preprocess(text) -> str:
    """
    Tiền xử lý nâng cao:
    - Thực hiện toàn bộ basic_preprocess
    - Tách từ tiếng Việt: ví dụ "sinh viên" -> "sinh_viên"
    - Xóa stopwords tiếng Việt
    """
    text = basic_preprocess(text)
    text = _word_segment(text)
    text = _remove_stopwords(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_preprocessor(method: str) -> Callable:
    """
    Lấy hàm tiền xử lý theo tên phương pháp.
    method gồm: none, basic, advanced.
    """
    preprocessors: dict[str, Callable] = {
        "none": none_preprocess,
        "basic": basic_preprocess,
        "advanced": advanced_preprocess,
    }

    method = method.lower().strip()

    if method not in preprocessors:
        raise ValueError(
            "Phương pháp tiền xử lý không hợp lệ! "
            "Chọn: 'none', 'basic', hoặc 'advanced'."
        )

    return preprocessors[method]


def apply_preprocessing(
    df: pd.DataFrame,
    method: str = "none",
    text_column: str = "Comment"
) -> pd.DataFrame:
    """
    Áp dụng một phương pháp tiền xử lý cho DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Dữ liệu đầu vào.

    method : str
        Một trong ba giá trị: 'none', 'basic', 'advanced'.

    text_column : str
        Tên cột chứa văn bản cần xử lý.

    Returns
    -------
    pd.DataFrame
        DataFrame gồm cột văn bản đã xử lý và các cột nhãn nếu tồn tại.
    """
    if text_column not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{text_column}' trong DataFrame!")

    processor = get_preprocessor(method)

    df_processed = df.copy()
    df_processed[text_column] = df_processed[text_column].apply(processor)

    columns_to_keep = [text_column]

    # Giữ lại các cột nhãn thường gặp nếu có
    for label_col in ["Label", "label", "SpamLabel", "spam_label", "Rating", "rating"]:
        if label_col in df_processed.columns and label_col not in columns_to_keep:
            columns_to_keep.append(label_col)

    return df_processed[columns_to_keep]


def find_text_column(df: pd.DataFrame) -> str:
    """
    Tự tìm cột chứa văn bản trong file CSV.
    Ưu tiên các tên cột thường gặp.
    """
    possible_columns = [
        "Comment", "comment",
        "Text", "text",
        "Message", "message",
        "Content", "content"
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    raise ValueError(
        "Không tìm thấy cột văn bản trong file CSV. "
        f"Các cột hiện có là: {list(df.columns)}"
    )


def export_preprocessed_csvs(
    input_csv_path: Path,
    output_dir: Path,
    text_column: str = "Comment"
) -> None:
    """
    Đọc toàn bộ file CSV đầu vào, áp dụng 3 kiểu preprocessing
    và xuất ra 3 file CSV riêng.

    File xuất ra:
    - test_none.csv
    - test_basic.csv
    - test_advanced.csv

    Trong mỗi file:
    - OriginalComment: câu gốc
    - Comment: câu sau khi xử lý, dùng trực tiếp cho TF-IDF
    - Các cột còn lại như Rating, Label, SpamLabel được giữ nguyên
    """
    if not input_csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_csv_path}")

    df = pd.read_csv(input_csv_path)

    if text_column not in df.columns:
        text_column = find_text_column(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessors: dict[str, Callable] = {
        "none": none_preprocess,
        "basic": basic_preprocess,
        "advanced": advanced_preprocess,
    }

    print("===== THÔNG TIN FILE GỐC =====")
    print("File:", input_csv_path)
    print("Số dòng, số cột:", df.shape)
    print("Các cột:", list(df.columns))
    print("Cột văn bản:", text_column)
    print()

    for method_name, processor_func in preprocessors.items():
        df_output = df.copy()

        # Lưu lại câu gốc
        original_text = df_output[text_column].copy()

        # Thay cột Comment bằng câu đã xử lý
        df_output[text_column] = df_output[text_column].apply(processor_func)

        # Chèn OriginalComment ngay trước cột Comment cho dễ so sánh
        comment_index = df_output.columns.get_loc(text_column)
        df_output.insert(comment_index, "OriginalComment", original_text)

        output_path = output_dir / f"test_{method_name}.csv"

        df_output.to_csv(output_path, index=False, encoding="utf-8-sig")

        print("=" * 80)
        print(f"Đã xuất file: {output_path}")
        print("Số dòng, số cột:", df_output.shape)
        print(df_output[["OriginalComment", text_column]].head(10))
        print()

if __name__ == "__main__":
    # File processor.py nằm trong thư mục core/
    # Muốn lấy data/test.csv thì đi ra ngoài 1 cấp rồi vào thư mục data/
    base_dir = Path(__file__).resolve().parent.parent

    input_csv_path = base_dir / "data" / "test.csv"
    output_dir = base_dir / "data" / "preprocessed"

    export_preprocessed_csvs(
        input_csv_path=input_csv_path,
        output_dir=output_dir,
        text_column="Comment"
    )