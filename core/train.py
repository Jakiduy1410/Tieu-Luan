from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import os

class SpamClassifier:
    def __init__(self, model_type='logistic_regression'):
        """
        Khởi tạo mô hình học máy.
        :param model_type: 'logistic_regression', 'naive_bayes', 'svm', 'random_forest'
        """
        self.model_type = model_type
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'svm':
            self.model = LinearSVC(max_iter=2000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError(f"Không hỗ trợ mô hình: {model_type}")

    def train(self, X_train, y_train):
        """Huấn luyện mô hình"""
        self.model.fit(X_train, y_train)
        print(f"✅ Đã huấn luyện xong mô hình: {self.model_type}")

    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình và trả về các chỉ số hiệu suất"""
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        return metrics

    def save_model(self, filepath):
        """Lưu mô hình ra file"""
        joblib.dump(self.model, filepath)
        print(f"Đã lưu mô hình tại: {filepath}")

    @classmethod
    def load_model(cls, filepath, model_type):
        """Tải mô hình từ file"""
        instance = cls(model_type=model_type)
        instance.model = joblib.load(filepath)
        return instance

def run_experiment(X_train, X_test, y_train, y_test, models_to_run=['naive_bayes', 'svm', 'random_forest']):
    """
    Hàm tiện ích để chạy nhanh nhiều mô hình cùng lúc và so sánh.
    """
    results = {}
    for m_type in models_to_run:
        print(f"\n--- Đang chạy mô hình: {m_type} ---")
        clf = SpamClassifier(model_type=m_type)
        clf.train(X_train, y_train)
        metrics = clf.evaluate(X_test, y_test)
        results[m_type] = metrics
        print(f"Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1_score']:.4f}")
    
    return results
