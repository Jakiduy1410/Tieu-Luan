# Phân loại Tin nhắn Spam (Vietnamese Spam Detection)

Dự án này là một hệ thống thử nghiệm nhằm đánh giá tác động của các kỹ thuật **Tiền xử lý văn bản (Text Preprocessing)** lên hiệu suất của các mô hình **Học máy (Machine Learning)** trong bài toán phân loại tin nhắn/bình luận tiếng Việt (Spam vs. Ham).

## 🗂️ Sơ đồ Khung Dự án (Project Structure)

```text
📦 Tieu Luan/
 ┣ 📂 data/
 ┃ ┣ 📜 train.csv          # Dữ liệu huấn luyện gốc
 ┃ ┗ 📜 test.csv           # Dữ liệu kiểm tra gốc
 ┣ 📂 preprocessing/       # Chứa các phương pháp tiền xử lý văn bản
 ┃ ┣ 📜 none.py            # Baseline: Không tiền xử lý (giữ nguyên gốc)
 ┃ ┣ 📜 basic.py           # Basic: Xóa dấu câu, viết thường, chuẩn hóa whitespace
 ┃ ┗ 📜 advanced.py        # Advanced: Tách từ (Word Segmentation), xóa Stopwords tiếng Việt
 ┣ 📂 core/                # Các thành phần cốt lõi của pipeline
 ┃ ┣ 📜 tfidf.py           # Module trích xuất đặc trưng văn bản bằng TF-IDF
 ┃ ┣ 📜 train.py           # Module huấn luyện các mô hình Machine Learning
 ┃ ┗ 📜 crawl.py           # (Chức năng mở rộng thu thập dữ liệu)
 ┣ 📜 main.py              # File thực thi toàn bộ pipeline (Từ raw data tới kết quả)
 ┗ 📜 README.md            # Tài liệu dự án
```

## 🎯 Hướng đi tiếp theo và Yêu cầu kỹ thuật chi tiết

Để dự án đạt kết quả nghiên cứu tốt nhất và tránh hiện tượng quá khớp (overfitting), các thành viên thực hiện cần tuân thủ các yêu cầu sau:

### 1. Module `preprocessing` (Tiền xử lý văn bản)
Cần hiện thực hóa 3 hàm trong file `processor.py` với các tiêu chuẩn sau:

*   **None Preprocessing (`none_preprocess`)**:
    *   Giữ nguyên văn bản gốc.
    *   Xử lý các giá trị `NaN` hoặc `None` thành chuỗi rỗng `""`.
*   **Basic Preprocessing (`basic_preprocess`)**:
    *   Chuyển toàn bộ văn bản về chữ thường (**Lowercase**).
    *   Loại bỏ toàn bộ dấu câu và các ký tự đặc biệt (**Punctuation removal**).
    *   Loại bỏ các chữ số.
    *   Chuẩn hóa khoảng trắng (xóa khoảng trắng thừa ở đầu, cuối và giữa các từ).
*   **Advanced Preprocessing (`advanced_preprocess`)**:
    *   Thực hiện toàn bộ các bước của Basic.
    *   **Tách từ tiếng Việt (Word Segmentation)**: Sử dụng thư viện `pyvi` hoặc `underthesea` để nối các từ ghép (ví dụ: `sinh viên` -> `sinh_viên`).
    *   **Loại bỏ Stopwords**: Sử dụng danh sách từ dừng tiếng Việt để lọc bỏ các từ không mang giá trị phân loại (ví dụ: "và", "là", "của", "thì"...).

### 2. Module `model` (Huấn luyện và Tối ưu hóa)
Để mô hình đạt hiệu suất cao nhất và bền vững, quy trình huấn luyện phải bao gồm bước **Tối ưu hóa tham số (Hyperparameter Tuning)**:

*   **Grid Search & Cross-Validation**:
    *   Trước khi huấn luyện chính thức, phải sử dụng `GridSearchCV` từ `sklearn` để tìm ra bộ tham số tốt nhất cho từng mô hình.
    *   Sử dụng **K-Fold Cross Validation** (thường là K=5) trong quá trình Grid Search để đảm bảo mô hình không bị Overfit trên một tập dữ liệu nhỏ.
*   **Các tham số cần tinh chỉnh**:
    *   **SVM**: Tinh chỉnh `C` (độ phạt), `kernel` (rbf, linear, poly), và `gamma`.
    *   **Random Forest**: Tinh chỉnh `n_estimators` (số lượng cây), `max_depth` (độ sâu tối đa), và `min_samples_split`.
    *   **Naive Bayes**: Tinh chỉnh tham số làm phẳng `alpha`.
*   **Đánh giá**:
    *   Sau khi tìm được "Best Estimator", mô hình sẽ được đánh giá trên tập **Test** độc lập chưa từng xuất hiện trong quá trình huấn luyện.

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ**: Python 3
- **Thư viện chính**: `pandas`, `numpy`, `scikit-learn`
- **Thư viện NLP**: `pyvi` hoặc `underthesea` (cho tiếng Việt)
- **Visualization**: `matplotlib`, `seaborn`

## Reproducible Research Experiment

Pipeline chinh cho bai nghien cuu da duoc cai dat trong `core/research_experiment.py`.
Script nay chay ca baseline va GridSearchCV theo dung quy trinh:

```bash
python run_research_experiment.py
```

Mac dinh script se:

- Doc `data/train.csv` de train va chon tham so.
- Doc `data/test.csv` chi de danh gia cuoi cung.
- Chay 3 kieu preprocessing: `none`, `basic`, `advanced`.
- Chay 4 model: `naive_bayes`, `linear_svm`, `random_forest`, `logistic_regression`.
- Dung `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Xuat ket qua vao `results/research_YYYYMMDD_HHMMSS/`.

Artifacts duoc sinh ra gom:

- `baseline_results.csv`
- `tuned_results.csv`
- `tuned_results.md`
- `all_results.csv`
- `best_confusion_matrix.csv`
- `best_classification_report.csv`
- `best_model_summary.md`
- `f1_macro_comparison.png`
- `best_pipeline.joblib`

Chay smoke test nhanh truoc khi chay full grid:

```bash
python run_research_experiment.py --quick --sample-size 300 --cv-folds 2 --preprocessing none,basic --models naive_bayes,linear_svm
```

Chay mot tap model/preprocessing cu the:

```bash
python run_research_experiment.py --preprocessing basic,advanced --models linear_svm,logistic_regression
```
