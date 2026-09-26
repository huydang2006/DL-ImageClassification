"""Entry point cho đánh giá model & sinh báo cáo. python evaluate.py --model M1|M2|M3"""

import argparse


def evaluate(model_name):
    """
    Load model đã huấn luyện từ models/, đánh giá trên tập test.

    Các bước (placeholder):
    1. load dataset test
    2. load weights từ models/<model_name>*.weights.h5
    3. predict trên test set
    4. Tính metrics: accuracy, precision, recall, F1 (overall + per-class)
    5. Vẽ confusion matrix, ROC curve -> lưu vào results/plots/
    6. Lưu bảng metrics -> results/metrics/<model_name>_metrics.json
    """
    return NotImplemented


def main():
    parser = argparse.ArgumentParser(description="Đánh giá & so sánh mô hình")
    parser.add_argument("--model", type=str, default="M1", choices=["M1", "M2", "M3"])
    args = parser.parse_args()
    evaluate(args.model)


if __name__ == "__main__":
    main()
