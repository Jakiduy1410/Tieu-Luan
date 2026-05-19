from pathlib import Path
import pandas as pd
import re
import html

# =========================
# PATH CONFIG
# =========================

from pathlib import Path

# Nếu file check_preprocessing_stats.py đang nằm trong thư mục core
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_DIR.mkdir(exist_ok=True)

TEXT_COLUMN_CANDIDATES = [
    "Comment", "comment",
    "Text", "text",
    "Review", "review",
    "content", "Content",
    "sentence", "Sentence"
]

# Stopwords đơn giản để mô phỏng mức advanced
VI_STOPWORDS = set("""
bị bởi cả các cái cần càng chỉ cho chứ chưa có cũng đã đang đây để đến đều điều
do đó được gì khi không là lại lên lúc mà một nên nếu nhiều như nhưng những nữa
phải qua ra rằng rất rồi sau sẽ sự tại theo thì trên trước từ và vẫn vào vậy vì
với vừa tôi bạn anh chị em mình họ nó này kia ấy đây đâu nào sao nhé nha
""".split())


# =========================
# HELPER FUNCTIONS
# =========================

def read_csv_safely(path: Path) -> pd.DataFrame:
    """Đọc CSV, thử nhiều encoding để tránh lỗi font."""
    encodings = ["utf-8-sig", "utf-8", "latin1"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(f"Không đọc được file: {path}")


def find_text_column(df: pd.DataFrame) -> str:
    """Tự tìm cột chứa văn bản."""
    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if not object_cols:
        raise ValueError("Không tìm thấy cột văn bản trong file CSV.")

    # Nếu không có tên cột quen thuộc, lấy cột text có độ dài trung bình lớn nhất
    best_col = max(
        object_cols,
        key=lambda c: df[c].fillna("").astype(str).str.len().mean()
    )

    return best_col


def preprocess_none(text):
    """Giữ gần như nguyên bản."""
    if pd.isna(text):
        return ""
    return str(text).strip()


def preprocess_basic(text):
    """Tiền xử lý cơ bản."""
    text = preprocess_none(text)
    text = text.lower()
    text = html.unescape(text)

    # Xóa URL, email, HTML tag
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    # Xóa dấu câu/ký tự đặc biệt, giữ chữ, số, dấu tiếng Việt
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_advanced(text):
    """Tiền xử lý nâng cao: basic + bỏ stopwords + bỏ số."""
    text = preprocess_basic(text)

    tokens = tokenize(text)
    tokens = [
        token for token in tokens
        if token not in VI_STOPWORDS and not token.isdigit()
    ]

    return " ".join(tokens)


def tokenize(text):
    """Tách token đơn giản."""
    if pd.isna(text):
        return []

    text = str(text)
    return re.findall(r"(?u)\b\w+\b", text)


PREPROCESSORS = {
    "none": preprocess_none,
    "basic": preprocess_basic,
    "advanced": preprocess_advanced
}


def load_or_process_texts(dataset_name: str, level: str):
    """
    Ưu tiên đọc file đã tiền xử lý trong data/preprocessed.
    Nếu không có thì tự xử lý lại từ train.csv/test.csv.
    """
    preprocessed_path = PREPROCESSED_DIR / f"{dataset_name}_{level}.csv"
    raw_path = DATA_DIR / f"{dataset_name}.csv"

    if preprocessed_path.exists():
        df = read_csv_safely(preprocessed_path)
        text_col = find_text_column(df)
        texts = df[text_col].fillna("").astype(str).apply(preprocess_none)
        source = str(preprocessed_path)
    else:
        df = read_csv_safely(raw_path)
        text_col = find_text_column(df)
        processor = PREPROCESSORS[level]
        texts = df[text_col].fillna("").astype(str).apply(processor)
        source = str(raw_path) + f" | tự xử lý mức {level}"

    return texts, source


def calculate_stats(dataset_name: str, level: str):
    texts, source = load_or_process_texts(dataset_name, level)

    token_lists = texts.apply(tokenize)

    avg_text_length = texts.apply(len).mean()
    avg_token_count = token_lists.apply(len).mean()
    vocab_size = len(set(token for tokens in token_lists for token in tokens))

    return {
        "Tập dữ liệu": dataset_name.capitalize(),
        "Mức tiền xử lý": level.capitalize(),
        "Độ dài văn bản trung bình": round(avg_text_length, 2),
        "Số token trung bình": round(avg_token_count, 2),
        "Số từ vựng khác nhau": vocab_size,
        "Nguồn dữ liệu": source
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Xuất markdown không cần thư viện tabulate."""
    headers = list(df.columns)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")

    return "\n".join(lines)


# =========================
# MAIN
# =========================

def main():
    rows = []

    for dataset_name in ["train", "test"]:
        for level in ["none", "basic", "advanced"]:
            rows.append(calculate_stats(dataset_name, level))

    stats_df = pd.DataFrame(rows)

    # Bản dùng cho báo cáo: bỏ cột nguồn dữ liệu
    report_df = stats_df.drop(columns=["Nguồn dữ liệu"])

    print("\nBẢNG THỐNG KÊ MÔ TẢ SAU CÁC MỨC TIỀN XỬ LÝ:\n")
    print(report_df.to_string(index=False))

    # Lưu file CSV đầy đủ
    csv_path = RESULTS_DIR / "preprocessing_descriptive_stats.csv"
    stats_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Lưu file Markdown để copy vào báo cáo
    md_path = RESULTS_DIR / "preprocessing_descriptive_stats.md"
    md_content = dataframe_to_markdown(report_df)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\nĐã lưu kết quả:")
    print(f"- {csv_path}")
    print(f"- {md_path}")


if __name__ == "__main__":
    main()