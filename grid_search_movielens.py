"""
grid_search_movielens.py -- Quét lưới tham số (k, min_support) cho KNNBasic
và KNNWithMeans trên dữ liệu MovieLens đọc từ DB, tìm tổ hợp cho kết quả tốt
nhất ở từng chỉ số (RMSE, MAE, Precision@K, Recall@K).

Tái sử dụng nguyên vẹn cơ chế đánh giá đã có trong evaluate_movielens.py
(cross_validate 5-fold cho RMSE/MAE, KFold + precision_recall_at_k chuẩn FAQ
surprise cho Precision/Recall@K) -- import trực tiếp, không viết lại logic.

Lưới quét: k thuộc {5,10,20,30,40,60}, min_support thuộc {2,3,5,10,15}
  = 30 tổ hợp x 2 thuật toán = 60 lần full-evaluate (mỗi lần gồm cross_validate
  5-fold RMSE/MAE + KFold 5-fold Precision/Recall@K).

Cách chọn "tốt nhất":
  - Với mỗi thuật toán riêng: tìm tổ hợp (k, min_support) cho từng chỉ số tốt
    nhất (RMSE thấp nhất, MAE thấp nhất, Precision@K cao nhất, Recall@K cao
    nhất) -- 4 tổ hợp "vô địch" theo 4 chỉ số, có thể trùng hoặc khác nhau.
  - So sánh chéo 2 thuật toán TẠI tổ hợp tham số tốt nhất của từng thuật toán:
    nếu KNNWithMeans tại tham số tối ưu của nó thắng TUYỆT ĐỐI cả 4 chỉ số so
    với KNNBasic tại tham số tối ưu của nó -> báo "thắng tuyệt đối". Ngược lại
    (trường hợp thường gặp vì RMSE/MAE và Precision/Recall có xu hướng ngược
    chiều -- xem ghi chú trong evaluate_movielens.py) -> liệt kê riêng tham số
    tốt nhất theo từng chỉ số, không ép ra 1 "người thắng" duy nhất.

Chạy: python grid_search_movielens.py [--k 10] [--threshold 4.0] [--cv 5]

Cảnh báo thời gian chạy: lưới mặc định ước tính có thể mất gần 2 giờ tùy máy.
"""

import argparse
import itertools
import json
import sys
import time
from datetime import datetime

from surprise import Dataset, KNNBasic, KNNWithMeans, Reader

sys.path.insert(0, ".")

from app.db.queries import load_all_reviews
from app.db.session import SessionLocal
from evaluate_movielens import fmt, run_precision_recall_kfold, run_rmse_mae

K_GRID = [5, 10, 20, 30, 40, 60]
MIN_SUPPORT_GRID = [2, 3, 5, 10, 15]

ALGORITHMS = [("KNNBasic", KNNBasic), ("KNNWithMeans", KNNWithMeans)]


def run_one_combo(algo_class, data, k, min_support, top_k, threshold, cv):
    sim_options = {"name": "cosine", "user_based": True, "min_support": min_support}
    algo_kwargs = {"k": k, "min_k": 2, "sim_options": sim_options}

    rmse_mae = run_rmse_mae(algo_class, data, cv=cv, **algo_kwargs)
    precision, recall = run_precision_recall_kfold(
        algo_class, data, top_k=top_k, threshold=threshold, n_splits=cv, **algo_kwargs
    )
    return {
        "k": k,
        "min_support": min_support,
        "rmse": rmse_mae["rmse_mean"],
        "mae": rmse_mae["mae_mean"],
        "precision": precision,
        "recall": recall,
        "fit_time": rmse_mae["fit_time_mean"],
    }


def best_by_metric(rows: list[dict], metric: str, higher_is_better: bool) -> dict:
    key = (lambda r: -r[metric]) if higher_is_better else (lambda r: r[metric])
    return min(rows, key=key)


def main():
    parser = argparse.ArgumentParser(
        description="Grid search (k, min_support) cho KNNBasic vs KNNWithMeans trên dữ liệu DB."
    )
    parser.add_argument("--k", type=int, default=10, help="K cho Precision@K/Recall@K (mặc định 10).")
    parser.add_argument("--threshold", type=float, default=4.0, help="Ngưỡng relevant rating (mặc định 4.0).")
    parser.add_argument("--cv", type=int, default=5, help="Số fold cho cross_validate và KFold (mặc định 5).")
    args = parser.parse_args()

    print("=== Grid search KNNBasic vs KNNWithMeans (k, min_support) ===")
    print(f"K_GRID={K_GRID}")
    print(f"MIN_SUPPORT_GRID={MIN_SUPPORT_GRID}")
    print(f"Precision/Recall@{args.k}, threshold={args.threshold}, cv={args.cv}")
    total_combos = len(K_GRID) * len(MIN_SUPPORT_GRID) * len(ALGORITHMS)
    print(f"Tổng số lần full-evaluate: {total_combos}\n")

    print("Đang đọc dữ liệu từ DB...")
    db = SessionLocal()
    try:
        review_df = load_all_reviews(db)
    finally:
        db.close()
    print(f"  {len(review_df)} rating, {review_df['user_id'].nunique()} user, "
          f"{review_df['movie_id'].nunique()} phim\n")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(review_df[["user_id", "movie_id", "rating"]], reader)

    all_results: dict[str, list[dict]] = {name: [] for name, _ in ALGORITHMS}

    t_start = time.time()
    combo_idx = 0
    for name, algo_class in ALGORITHMS:
        for k, min_support in itertools.product(K_GRID, MIN_SUPPORT_GRID):
            combo_idx += 1
            t0 = time.time()
            row = run_one_combo(algo_class, data, k, min_support, args.k, args.threshold, args.cv)
            elapsed = time.time() - t0
            all_results[name].append(row)
            print(
                f"[{combo_idx}/{total_combos}] {name} k={k} min_support={min_support} "
                f"-> RMSE={fmt(row['rmse'])} MAE={fmt(row['mae'])} "
                f"P@{args.k}={fmt(row['precision'])} R@{args.k}={fmt(row['recall'])} "
                f"({elapsed:.1f}s)"
            )

    total_elapsed = time.time() - t_start
    print(f"\nTổng thời gian grid search: {total_elapsed / 60:.1f} phút\n")

    # ---- Tìm tham số tốt nhất theo từng chỉ số, cho từng thuật toán ----
    best_per_algo: dict[str, dict] = {}
    for name in all_results:
        rows = all_results[name]
        best_per_algo[name] = {
            "rmse": best_by_metric(rows, "rmse", higher_is_better=False),
            "mae": best_by_metric(rows, "mae", higher_is_better=False),
            "precision": best_by_metric(rows, "precision", higher_is_better=True),
            "recall": best_by_metric(rows, "recall", higher_is_better=True),
        }

    lines = []
    lines.append("=" * 78)
    lines.append("  GRID SEARCH: THAM SỐ TỐT NHẤT THEO TỪNG CHỈ SỐ")
    lines.append("=" * 78)
    lines.append("")

    for name in all_results:
        lines.append(f"--- {name} ---")
        for metric, label in [
            ("rmse", "RMSE thấp nhất"), ("mae", "MAE thấp nhất"),
            ("precision", f"Precision@{args.k} cao nhất"), ("recall", f"Recall@{args.k} cao nhất"),
        ]:
            b = best_per_algo[name][metric]
            lines.append(
                f"  {label:<28}: k={b['k']:<3} min_support={b['min_support']:<3} "
                f"-> RMSE={fmt(b['rmse'])} MAE={fmt(b['mae'])} "
                f"P@{args.k}={fmt(b['precision'])} R@{args.k}={fmt(b['recall'])}"
            )
        lines.append("")

    # ---- So sánh chéo: KNNWithMeans tại tham số tối ưu riêng vs KNNBasic tại
    # tham số tối ưu riêng -- thắng tuyệt đối cả 4 chỉ số hay không ----
    lines.append("=" * 78)
    lines.append("  SO SÁNH CHÉO: KNNBasic (tối ưu riêng) vs KNNWithMeans (tối ưu riêng)")
    lines.append("=" * 78)
    lines.append("")

    # Với mỗi chỉ số, so trực tiếp giá trị tốt nhất mà mỗi thuật toán đạt được
    # (không cùng 1 tổ hợp tham số -- đây là so sánh "khả năng tốt nhất có thể
    # đạt" của từng thuật toán trên toàn lưới, không phải so tại 1 điểm cố định).
    metrics_direction = [
        ("rmse", False, "RMSE"), ("mae", False, "MAE"),
        ("precision", True, f"Precision@{args.k}"), ("recall", True, f"Recall@{args.k}"),
    ]

    wins = {"KNNBasic": 0, "KNNWithMeans": 0}
    detail_lines = []
    for metric, higher_better, label in metrics_direction:
        v_basic = best_per_algo["KNNBasic"][metric][metric]
        v_means = best_per_algo["KNNWithMeans"][metric][metric]
        if higher_better:
            winner = "KNNWithMeans" if v_means > v_basic else ("KNNBasic" if v_basic > v_means else "Hòa")
        else:
            winner = "KNNWithMeans" if v_means < v_basic else ("KNNBasic" if v_basic < v_means else "Hòa")
        if winner in wins:
            wins[winner] += 1
        detail_lines.append(
            f"  {label:<15}: KNNBasic best={fmt(v_basic)}  KNNWithMeans best={fmt(v_means)}  -> {winner} thắng"
        )

    lines.extend(detail_lines)
    lines.append("")

    if wins["KNNWithMeans"] == 4:
        lines.append(">> KNNWithMeans THẮNG TUYỆT ĐỐI cả 4 chỉ số (ở tham số tối ưu riêng của từng thuật toán).")
        lines.append(f">> Tham số đề xuất dùng chung: xem mục KNNWithMeans ở trên, ưu tiên tổ hợp")
        lines.append(f">>   xuất hiện nhiều nhất trong 4 dòng 'tốt nhất' (đồng thuận nhiều chỉ số).")
    elif wins["KNNBasic"] == 4:
        lines.append(">> KNNBasic THẮNG TUYỆT ĐỐI cả 4 chỉ số (ở tham số tối ưu riêng của từng thuật toán).")
    else:
        lines.append(
            f">> KHÔNG có thuật toán nào thắng tuyệt đối cả 4 chỉ số "
            f"(KNNBasic thắng {wins['KNNBasic']}/4, KNNWithMeans thắng {wins['KNNWithMeans']}/4)."
        )
        lines.append(">> Không ép ra 1 'người thắng' duy nhất -- dùng bảng tham số tốt nhất theo")
        lines.append(">> từng chỉ số ở trên để chọn tùy theo mục tiêu ưu tiên của hệ thống")
        lines.append(">> (ranking top-K -> ưu tiên Precision/Recall; dự đoán điểm -> ưu tiên RMSE/MAE).")

    output = "\n".join(lines)
    print(output)

    # ---- Ghi kết quả đầy đủ (raw + summary) ----
    out_txt = "grid_search_results.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\n>> Tóm tắt đã ghi vào {out_txt}")

    out_json = "grid_search_raw.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "k_grid": K_GRID,
                "min_support_grid": MIN_SUPPORT_GRID,
                "top_k": args.k,
                "threshold": args.threshold,
                "cv": args.cv,
                "n_ratings": len(review_df),
                "n_users": int(review_df["user_id"].nunique()),
                "n_movies": int(review_df["movie_id"].nunique()),
                "results": all_results,
                "best_per_algo": best_per_algo,
                "total_elapsed_seconds": total_elapsed,
            },
            f,
            indent=2,
            default=float,
        )
    print(f">> Dữ liệu thô (toàn bộ {total_combos} tổ hợp) đã ghi vào {out_json}")


if __name__ == "__main__":
    main()
