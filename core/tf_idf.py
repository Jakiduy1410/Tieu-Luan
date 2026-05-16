import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    def __init__(
        self,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    ):
        """
        Khởi tạo bộ biến đổi TF-IDF.

        Parameters
        ----------
        max_features : int hoặc None
            Số lượng đặc trưng tối đa.
            Ví dụ: 5000 nghĩa là chỉ lấy 5000 từ/cụm từ quan trọng nhất.

        ngram_range : tuple
            (1, 1): chỉ lấy từ đơn.
            (1, 2): lấy cả từ đơn và cụm 2 từ.

        min_df : int hoặc float
            Bỏ các từ xuất hiện quá ít.
            Ví dụ min_df=2 nghĩa là từ phải xuất hiện ít nhất trong 2 văn bản.

        max_df : float
            Bỏ các từ xuất hiện quá nhiều.
            Ví dụ max_df=0.95 nghĩa là bỏ từ xuất hiện trong hơn 95% văn bản.

        sublinear_tf : bool
            Nếu True, giảm ảnh hưởng của các từ lặp lại quá nhiều.
        """

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf
        )

    def _prepare_text_data(self, text_data):
        """
        Chuẩn hóa dữ liệu đầu vào trước khi đưa vào TF-IDF.
        Xử lý NaN/None và ép toàn bộ về string.
        """
        if isinstance(text_data, pd.Series):
            return text_data.fillna("").astype(str)

        if isinstance(text_data, list):
            return ["" if text is None else str(text) for text in text_data]

        return text_data

    def fit_transform(self, text_data):
        """
        Học từ vựng từ tập Train và chuyển đổi văn bản thành ma trận TF-IDF.

        Chỉ dùng hàm này cho tập train.
        Không dùng fit_transform cho tập test để tránh data leakage.
        """
        text_data = self._prepare_text_data(text_data)
        return self.vectorizer.fit_transform(text_data)

    def transform(self, text_data):
        """
        Chuyển đổi tập Test bằng bộ từ vựng đã học từ tập Train.

        Lưu ý:
        Không fit lại trên tập test.
        """
        text_data = self._prepare_text_data(text_data)
        return self.vectorizer.transform(text_data)

    def get_feature_names(self):
        """
        Trả về danh sách từ/cụm từ mà TF-IDF đã học được.
        """
        return self.vectorizer.get_feature_names_out()

    def save_model(self, filepath="models/tfidf_vectorizer.pkl"):
        """
        Lưu lại TF-IDF Vectorizer để tái sử dụng.
        """
        dir_path = os.path.dirname(filepath)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        joblib.dump(self.vectorizer, filepath)
        print(f"Đã lưu TF-IDF Vectorizer tại: {filepath}")

    @classmethod
    def load_model(cls, filepath="models/tfidf_vectorizer.pkl"):
        """
        Tải TF-IDF Vectorizer đã lưu.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

        instance = cls()
        instance.vectorizer = joblib.load(filepath)

        return instance


if __name__ == "__main__":
    sample_texts = [
        "sản phẩm rất tốt",
        "sản phẩm này quá tệ không nên mua",
        "tuyệt vời",
        "",
        None
    ]

    print("===== DỮ LIỆU GỐC =====")
    for text in sample_texts:
        print("-", text)

    print("\n===== KHỞI TẠO TF-IDF =====")
    vectorizer = TextVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )

    print("\n===== FIT_TRANSFORM =====")
    matrix = vectorizer.fit_transform(sample_texts)

    print("\n===== TỪ VỰNG ĐÃ HỌC =====")
    print(vectorizer.get_feature_names())

    print("\n===== KÍCH THƯỚC MA TRẬN TF-IDF =====")
    print(matrix.shape)

    print("\n===== VECTOR TF-IDF CỦA CÂU ĐẦU TIÊN =====")
    print(matrix.toarray()[0])

    print("\n===== LƯU MODEL TF-IDF =====")
    vectorizer.save_model("models/tfidf_vectorizer.pkl")