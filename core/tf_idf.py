import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

class TextVectorizer:
    def __init__(self, max_features=None, ngram_range=(1, 1)):
        """
        Khởi tạo bộ biến đổi TF-IDF.
        :param max_features: Số lượng từ tối đa (nhiều quá sẽ chậm mô hình, nếu để None là lấy toàn bộ).
        :param ngram_range: Lấy từ đơn (1, 1) hay cụm 2 từ (1, 2)
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit_transform(self, text_data):
        """
        Học từ vựng từ tập Train và chuyển đổi văn bản thành ma trận số (đặc trưng TF-IDF).
        :param text_data: Danh sách các câu văn bản (ví dụ: df_train['Comment'])
        :return: Ma trận TF-IDF
        """
        # Nếu truyền vào Series Pandas thì ép kiểu về list/array
        return self.vectorizer.fit_transform(text_data)

    def transform(self, text_data):
        """
        Sử dụng từ vựng đã học để chuyển đổi tập Test. 
        (Chú ý: Không dùng fit_transform cho tập test để tránh rò rỉ dữ liệu - data leakage)
        """
        return self.vectorizer.transform(text_data)

    def save_model(self, filepath='tfidf_vectorizer.pkl'):
        """Lưu lại mô hình TF-IDF để sau này tái sử dụng mà không cần fit lại"""
        joblib.dump(self.vectorizer, filepath)
        print(f"Đã lưu TF-IDF Vectorizer tại: {filepath}")

    @classmethod
    def load_model(cls, filepath='tfidf_vectorizer.pkl'):
        """Tải mô hình TF-IDF đã lưu"""
        if os.path.exists(filepath):
            instance = cls()
            instance.vectorizer = joblib.load(filepath)
            return instance
        else:
            raise FileNotFoundError(f"Không tìm thấy file {filepath}")

if __name__ == "__main__":
    # Test thử module tfidf
    sample_texts = [
        "Sản phẩm rất tốt",
        "Sản phẩm này quá tệ, không nên mua",
        "Tuyệt vời"
    ]
    
    print("--- Dữ liệu gốc ---")
    for txt in sample_texts:
        print(f"- {txt}")
        
    print("\nKhởi tạo TF-IDF Vectorizer...")
    vectorizer = TextVectorizer()
    
    print("\nThực hiện fit_transform...")
    matrix = vectorizer.fit_transform(sample_texts)
    
    print("\nTừ vựng (Vocabulary) đã học được:")
    print(vectorizer.vectorizer.vocabulary_)
    
    print("\nMa trận đầu ra (số dòng, số từ vựng):")
    print(matrix.shape)
    
    print("\nTF-IDF của câu đầu tiên ('Sản phẩm rất tốt'):")
    print(matrix.toarray()[0])
