from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. Xác định đường dẫn dự án
# =========================
# File này đặt trong thư mục core/
# Nên ROOT sẽ là thư mục TIEU-LUAN/
ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"

OUTPUT_DIR = ROOT / "results" / "label_distribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Hàm tìm cột nhãn
# =========================
def find_label_column(df: pd.DataFrame) -> str:
    possible_cols = ["Label", "label", "target", "class", "spam"]

    for col in possible_cols:
        if col in df.columns:
            return col

    raise ValueError(
        f"Không tìm thấy cột nhãn. Các cột hiện có là: {list(df.columns)}"
    )


# =========================
# 3. Hàm thống kê nhãn
# =========================
def get_label_stats(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    counts = df[label_col].value_counts().reindex([0, 1], fill_value=0)
    total = counts.sum()
    percentages = counts / total * 100

    stats = pd.DataFrame({
        "Nhãn": ["Ham (0)", "Spam (1)"],
        "Số lượng": [counts[0], counts[1]],
        "Tỷ lệ (%)": [percentages[0], percentages[1]]
    })

    return stats


# =========================
# 4. Hàm vẽ biểu đồ tròn
# =========================
def plot_pie_chart(df: pd.DataFrame, label_col: str, title: str, output_path: Path):
    counts = df[label_col].value_counts().reindex([0, 1], fill_value=0)

    labels = [
        f"Ham (0)\n{counts[0]} mẫu",
        f"Spam (1)\n{counts[1]} mẫu"
    ]

    plt.figure(figsize=(7, 7))
    plt.pie(
        counts.values,
        labels=labels,
        autopct="%1.2f%%",
        startangle=90
    )
    plt.title(title, fontsize=14)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# 5. Hàm vẽ chung train/test thành 1 hình
# =========================
def plot_combined_pie_chart(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_label_col: str,
    test_label_col: str,
    output_path: Path
):
    train_counts = train_df[train_label_col].value_counts().reindex([0, 1], fill_value=0)
    test_counts = test_df[test_label_col].value_counts().reindex([0, 1], fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    train_labels = [
        f"Ham (0)\n{train_counts[0]} mẫu",
        f"Spam (1)\n{train_counts[1]} mẫu"
    ]

    test_labels = [
        f"Ham (0)\n{test_counts[0]} mẫu",
        f"Spam (1)\n{test_counts[1]} mẫu"
    ]

    axes[0].pie(
        train_counts.values,
        labels=train_labels,
        autopct="%1.2f%%",
        startangle=90
    )
    axes[0].set_title("Tập train")
    axes[0].axis("equal")

    axes[1].pie(
        test_counts.values,
        labels=test_labels,
        autopct="%1.2f%%",
        startangle=90
    )
    axes[1].set_title("Tập test")
    axes[1].axis("equal")

    fig.suptitle("Hình 4.1. Biểu đồ phân bố số lượng mẫu theo nhãn", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# 6. Chạy chương trình
# =========================
def main():
    print("Đang đọc dữ liệu...")
    print(f"Train path: {TRAIN_PATH}")
    print(f"Test path : {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_label_col = find_label_column(train_df)
    test_label_col = find_label_column(test_df)

    print(f"\nCột nhãn train: {train_label_col}")
    print(f"Cột nhãn test : {test_label_col}")

    train_stats = get_label_stats(train_df, train_label_col)
    test_stats = get_label_stats(test_df, test_label_col)

    print("\n===== THỐNG KÊ TẬP TRAIN =====")
    print(train_stats.to_string(index=False))
    print(f"Tổng số mẫu train: {len(train_df)}")

    print("\n===== THỐNG KÊ TẬP TEST =====")
    print(test_stats.to_string(index=False))
    print(f"Tổng số mẫu test: {len(test_df)}")

    # Lưu bảng thống kê ra CSV
    train_stats.to_csv(OUTPUT_DIR / "label_distribution_train.csv", index=False, encoding="utf-8-sig")
    test_stats.to_csv(OUTPUT_DIR / "label_distribution_test.csv", index=False, encoding="utf-8-sig")

    # Vẽ biểu đồ riêng từng tập
    plot_pie_chart(
        train_df,
        train_label_col,
        "Phân bố nhãn trong tập train",
        OUTPUT_DIR / "hinh_4_1a_phan_bo_nhan_train.png"
    )

    plot_pie_chart(
        test_df,
        test_label_col,
        "Phân bố nhãn trong tập test",
        OUTPUT_DIR / "hinh_4_1b_phan_bo_nhan_test.png"
    )

    # Vẽ chung thành 1 hình để đưa vào báo cáo
    plot_combined_pie_chart(
        train_df,
        test_df,
        train_label_col,
        test_label_col,
        OUTPUT_DIR / "hinh_4_1_phan_bo_nhan_train_test.png"
    )

    print("\nĐã lưu kết quả vào thư mục:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()