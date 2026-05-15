import pandas as pd
import numpy as np

def none_preprocess(text):
    """
    Giữ nguyên văn bản gốc, chỉ xử lý lỗi NaN.
    """
    if pd.isna(text) or text is None:
        return ""
    return str(text)

def basic_preprocess(text):
    """
    Tiền xử lý cơ bản (Sẽ cập nhật code sau).
    """
    # Hiện tại tạm thời trả về giống None
    return none_preprocess(text)

def advanced_preprocess(text):
    """
    Tiền xử lý nâng cao (Sẽ cập nhật code sau).
    """
    # Hiện tại tạm thời trả về giống None
    return none_preprocess(text)

def apply_preprocessing(df, method='none', text_column='Comment'):
    """
    Hàm tổng quát để áp dụng các mức độ tiền xử lý.
    :param method: 'none', 'basic', hoặc 'advanced'
    """
    df_processed = df.copy()
    
    if method == 'none':
        func = none_preprocess
    elif method == 'basic':
        func = basic_preprocess
    elif method == 'advanced':
        func = advanced_preprocess
    else:
        raise ValueError("Phương pháp tiền xử lý không hợp lệ!")

    # Áp dụng hàm xử lý
    df_processed[text_column] = df_processed[text_column].apply(func)
    
    # Chỉ giữ lại cột Comment và Label cho gọn
    columns_to_keep = [text_column]
    if 'Label' in df.columns:
        columns_to_keep.append('Label')
        
    return df_processed[columns_to_keep]

if __name__ == "__main__":
    # Test nhanh
    test_data = pd.DataFrame({'Comment': ['Sản phẩm tốt', np.nan], 'Label': [0, 0]})
    print("Test None Preprocessing:")
    print(apply_preprocessing(test_data, method='none'))
