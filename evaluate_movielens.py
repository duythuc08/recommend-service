"""
evaluate_movielens.py -- So sánh KNNBasic vs KNNWithMeans trên dữ liệu MovieLens
đã import vào DB của hệ thống.

Thay thế cho phần so sánh "CF thuần vs CF + Implicit" trong evaluate.py
(đã tạm ngưng cùng implicit_scoring.py). Mục tiêu bây giờ là so sánh 2
biến thể của thuật toán User-Based CF trong cùng thư viện scikit-surprise:

  - KNNBasic       : r_hat(u,i) = sum_v sim(u,v)*r(v,i) / sum_v sim(u,v)
                     (KHÔNG mean-centering)
  - KNNWithMeans    : r_hat(u,i) = mean_u + sum_v sim(u,v)*(r(v,i)-mean_v) / sum_v sim(u,v)
                     (CÓ mean-centering, đây là thuật toán hệ thống đang dùng)

Dữ liệu đọc THẲNG từ DB (bảng `review`, qua load_all_reviews() -- data access
layer sẵn có của recommend-service, dùng đúng 1 query, không N+1) thay vì gọi
Dataset.load_builtin(). DB trỏ tới (REC_DB_NAME trong .env) đã được import sẵn
MovieLens 100k (~100k rating, ~1682 phim, ~945 user) -- xem README/ghi chú vận
hành riêng cho bước import này, không thuộc phạm vi script.

Phương pháp đánh giá GIỮ NGUYÊN như thiết kế ban đầu (xem ghi chú "Căn cứ
phương pháp" cuối file) dù nguồn dữ liệu đổi từ load_builtin() sang DB -- cả
hai đều cho ra cùng 1 kiểu đối tượng surprise.Dataset nên cơ chế chia train/
test không đổi:
  - RMSE / MAE      : cross_validate 5-fold (trung bình + độ lệch chuẩn qua
                       các fold) -- đúng cách khuyến nghị của surprise thay
                       vì 1 lần train_test_split ngẫu nhiên.
  - Precision@K,
    Recall@K        : theo đúng công thức mẫu chính thức trong FAQ của
                       scikit-surprise (precision_recall_at_k), chạy trên
                       KFold(n_splits=5), threshold mặc định = 4.0,
                       k mặc định = 10. Relevant = true rating >= threshold;
                       Recommended = nằm trong top-K theo điểm dự đoán VÀ
                       điểm dự đoán >= threshold.

Chạy: python evaluate_movielens.py [--k 10] [--threshold 4.0] [--cv 5]
"""

import argparse
import sys
import time
from collections import defaultdict

from surprise import Dataset, KNNBasic, KNNWithMeans, Reader
from surprise.model_selection import KFold, cross_validate

sys.path.insert(0, ".")

from app.db.queries import load_all_reviews
from app.db.session import SessionLocal


# ---------------------------------------------------------------------------
# Precision@K / Recall@K -- theo đúng mẫu chính thức trong FAQ của surprise
# (https://surprise.readthedocs.io/en/stable/FAQ.html), giữ nguyên logic gốc.
# ---------------------------------------------------------------------------
def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """
    Trả về dict[uid -> (precision, recall)] cho một tập predictions
    (kết quả của algo.test(testset)).

    Relevant item  = true rating (r_ui) >= threshold.
    Recommended item = nằm trong top-K theo est (điểm dự đoán) VÀ est >= threshold.
    """
    user_est_true = defaultdict(list)
    for uid, _iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}
    for uid, user_ratings in user_est_true.items():
        # Sắp theo điểm dự đoán giảm dần
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(
            (true_r >= threshold) and (est >= threshold)
            for (est, true_r) in user_ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def run_precision_recall_kfold(algo_class, data, top_k, threshold, n_splits=5, **algo_kwargs):
    """
    Chạy Precision@K/Recall@K trên KFold(n_splits), lấy trung bình precision
    và recall qua toàn bộ user và toàn bộ fold -- đúng mẫu FAQ của surprise.

    top_k: K của Precision@K/Recall@K (số item gợi ý xét đến) -- đặt tên khác
    với 'k' trong algo_kwargs (số láng giềng KNN) để tránh xung đột tham số
    khi unpack **algo_kwargs vào lệnh gọi algo_class(**algo_kwargs).
    """
    kf = KFold(n_splits=n_splits, random_state=42)
    all_precisions = []
    all_recalls = []

    for trainset, testset in kf.split(data):
        algo = algo_class(**algo_kwargs)
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=top_k, threshold=threshold)

        all_precisions.extend(precisions.values())
        all_recalls.extend(recalls.values())

    avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else float("nan")
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else float("nan")
    return avg_precision, avg_recall


# ---------------------------------------------------------------------------
# RMSE / MAE -- cross_validate 5-fold
# ---------------------------------------------------------------------------
def run_rmse_mae(algo_class, data, cv=5, **algo_kwargs):
    algo = algo_class(**algo_kwargs)
    t0 = time.time()
    results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=cv, verbose=False)
    elapsed = time.time() - t0
    return {
        "rmse_mean": results["test_rmse"].mean(),
        "rmse_std": results["test_rmse"].std(),
        "mae_mean": results["test_mae"].mean(),
        "mae_std": results["test_mae"].std(),
        "fit_time_mean": sum(results["fit_time"]) / len(results["fit_time"]),
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fmt(v, decimals=4):
    try:
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    parser = argparse.ArgumentParser(
        description="So sánh KNNBasic vs KNNWithMeans trên dữ liệu đọc thẳng từ DB (bảng review)."
    )
    parser.add_argument("--k", type=int, default=10, help="K cho Precision@K/Recall@K (mặc định 10).")
    parser.add_argument("--threshold", type=float, default=4.0, help="Ngưỡng relevant rating (mặc định 4.0).")
    parser.add_argument("--cv", type=int, default=5, help="Số fold cho cross_validate và KFold (mặc định 5).")
    args = parser.parse_args()

    print("=== So sánh KNNBasic vs KNNWithMeans (dữ liệu đọc từ DB, bảng review) ===")
    print(f"k={args.k}, threshold={args.threshold}, cv={args.cv}\n")

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

    # Cấu hình giống hệ thống đang dùng (cf_engine.py): cosine, user_based=True,
    # min_support=2, k=20 láng giềng.
    sim_options = {"name": "cosine", "user_based": True, "min_support": 2}
    algo_kwargs = {"k": 20, "min_k": 2, "sim_options": sim_options}

    results = {}
    for name, algo_class in [("KNNBasic", KNNBasic), ("KNNWithMeans", KNNWithMeans)]:
        print(f"--- {name}: RMSE/MAE (cross_validate, cv={args.cv}) ---")
        rmse_mae = run_rmse_mae(algo_class, data, cv=args.cv, **algo_kwargs)
        print(
            f"  RMSE = {fmt(rmse_mae['rmse_mean'])} (+/- {fmt(rmse_mae['rmse_std'])})  "
            f"MAE = {fmt(rmse_mae['mae_mean'])} (+/- {fmt(rmse_mae['mae_std'])})  "
            f"fit_time_tb = {fmt(rmse_mae['fit_time_mean'], 2)}s"
        )

        print(f"--- {name}: Precision@{args.k}/Recall@{args.k} (KFold, cv={args.cv}) ---")
        precision, recall = run_precision_recall_kfold(
            algo_class, data, top_k=args.k, threshold=args.threshold, n_splits=args.cv, **algo_kwargs
        )
        print(f"  Precision@{args.k} = {fmt(precision)}  Recall@{args.k} = {fmt(recall)}\n")

        results[name] = {**rmse_mae, "precision": precision, "recall": recall}

    # ---- Bảng so sánh ----
    col_w = 28
    lines = [
        "=" * 70,
        "  KẾT QUẢ SO SÁNH: KNNBasic vs KNNWithMeans (dữ liệu từ DB)",
        "=" * 70,
        "",
        f"{'Chỉ số':<{col_w}} {'KNNBasic':>18} {'KNNWithMeans':>18}",
        f"{'-'*col_w} {'-'*18} {'-'*18}",
        f"{'RMSE':<{col_w}} {fmt(results['KNNBasic']['rmse_mean']):>18} {fmt(results['KNNWithMeans']['rmse_mean']):>18}",
        f"{'MAE':<{col_w}} {fmt(results['KNNBasic']['mae_mean']):>18} {fmt(results['KNNWithMeans']['mae_mean']):>18}",
        f"{f'Precision@{args.k}':<{col_w}} {fmt(results['KNNBasic']['precision']):>18} {fmt(results['KNNWithMeans']['precision']):>18}",
        f"{f'Recall@{args.k}':<{col_w}} {fmt(results['KNNBasic']['recall']):>18} {fmt(results['KNNWithMeans']['recall']):>18}",
        f"{'Fit time TB (s)':<{col_w}} {fmt(results['KNNBasic']['fit_time_mean'], 2):>18} {fmt(results['KNNWithMeans']['fit_time_mean'], 2):>18}",
        f"{'-'*col_w} {'-'*18} {'-'*18}",
    ]
    output = "\n".join(lines)
    print(output)

    out_path = "evaluation_results_movielens.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\n>> Kết quả đã ghi vào {out_path}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Căn cứ phương pháp (research trước khi viết script này):
#
# - RMSE/MAE nên dùng k-fold cross-validation (surprise.model_selection.
#   cross_validate) thay vì 1 lần train_test_split ngẫu nhiên, để số liệu
#   ổn định và giảm phương sai do cách chia ngẫu nhiên riêng lẻ.
#   Nguồn: surprise getting_started.rst.
#
# - Precision@K/Recall@K dùng đúng công thức mẫu chính thức trong FAQ của
#   surprise (precision_recall_at_k), chạy trên KFold: relevant = true
#   rating >= threshold (mặc định 4.0), recommended = top-K theo est VÀ
#   est >= threshold. Nguồn: surprise FAQ.rst,
#   examples/precision_recall_at_k.py (Nicolas Hug).
#
# - Random KFold (thay vì Leave-One-Out hoặc temporal split) được chọn vì
#   đây chính là phương pháp trong ví dụ chính thức của thư viện, và vì cả
#   2 thuật toán (KNNBasic, KNNWithMeans) chạy trên CÙNG fold/seed nên vẫn
#   đảm bảo so sánh công bằng. Temporal split hoặc Leave-One-Out phản ánh
#   sát tình huống thực tế hơn (tận dụng timestamp có sẵn trong MovieLens)
#   nhưng phức tạp hơn -- có thể bổ sung sau nếu cần nâng cao độ chính xác
#   phương pháp luận.
#
# - KNNBasic KHÔNG trừ mean của từng user, KNNWithMeans CÓ mean-centering
#   (trừ mean rồi cộng lại mean của user đích). Mean-centering khử được
#   thiên lệch giữa các user (người khó tính chấm thấp, người dễ tính chấm
#   cao), nên KNNWithMeans thường cho RMSE/MAE thấp hơn (tốt hơn) KNNBasic.
#   Nguồn: surprise knn_inspired.rst; Ricci, Rokach & Shapira (2015),
#   Recommender Systems Handbook.
#
# - Nguồn dữ liệu: ban đầu script gọi Dataset.load_builtin('ml-100k') để có
#   1 bộ benchmark độc lập. Sau đó đổi sang đọc thẳng từ DB (load_all_reviews()
#   trên bảng `review`) vì DB cấu hình trong .env (REC_DB_NAME) đã được import
#   sẵn dữ liệu MovieLens 100k. Dataset.load_from_df() + Reader(rating_scale=
#   (1,5)) trả về CÙNG kiểu đối tượng surprise.Dataset như load_builtin(), nên
#   toàn bộ cơ chế chia train/test (cross_validate, KFold) không đổi.
# ---------------------------------------------------------------------------
