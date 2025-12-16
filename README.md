
# Credit Card Customer Churn Prediction

Bài Lab 2 (Numpy for Data Science) tập trung xây dựng quy trình dự đoán khách hàng rời bỏ dịch vụ thẻ tín dụng (customer churn) bằng **Logistic Regression cài đặt từ đầu chỉ với NumPy**. 

Dự án bao gồm 3 notebook theo pipeline: khám phá dữ liệu (EDA) → tiền xử lý & tạo tập train/valid/test → huấn luyện, tuning và đánh giá mô hình.

---

## 0. Table of Contents

- [1. Project Title & Description](#1-project-title--description)
- [2. Introduction](#2-introduction)
- [3. Dataset](#3-dataset)
- [4. Methodology (Important)](#4-methodology-important)
- [6. Installation & Setup](#6-installation--setup)
- [7. Usage](#7-usage)
- [8. Results](#8-results)
- [9. Project Structure](#9-project-structure)
- [10. Challenges & Solutions](#10-challenges--solutions)
- [11. Future Improvements](#11-future-improvements)
- [12. Contributors](#12-contributors)
- [13. License](#13-license)

---

## 1. Mô tả đồ án

**Credit Card Customer Churn Prediction**

Mục tiêu của dự án là xây dựng mô hình phân loại nhị phân dự đoán biến mục tiêu `Attrition_Flag` (khách hàng còn sử dụng dịch vụ hay đã rời bỏ) dựa trên các đặc trưng giao dịch và hành vi khách hàng. Toàn bộ quy trình được triển khai theo hướng **vectorization** trong NumPy để đảm bảo hiệu năng và tính nhất quán.

---

## 2. Giới thiệu

- **Vấn đề:** Trong lĩnh vực ngân hàng, việc khách hàng rời bỏ (attrition/churn) ảnh hưởng trực tiếp đến doanh thu và hiệu quả kinh doanh.
- **Động lực:** Chi phí giữ chân khách hàng hiện tại thường thấp hơn đáng kể so với chi phí thu hút khách hàng mới.
- **Mục tiêu:** Xây dựng mô hình phân loại nhị phân dự đoán `Attrition_Flag` và đánh giá bằng các chỉ số phù hợp (đặc biệt là **F1-score** do dữ liệu mất cân bằng).

---

## 3. Tập dữ liệu

- **Source:** [Kaggle - Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- **Target:** `Attrition_Flag` với hai lớp `Existing Customer` và `Attrited Customer`.
- **Kích thước dữ liệu:** 10,127 dòng và 23 cột (bao gồm 2 cột Naive Bayes được loại bỏ trong tiền xử lý).
- **Mất cân bằng lớp:** `Existing Customer` = 8,500 (83.93%), `Attrited Customer` = 1,627 (16.07%).
- **Một số đặc trưng quan trọng:** `Total_Trans_Ct`, `Total_Revolving_Bal`, `Contacts_Count_12_mon`, `Total_Trans_Amt`, ...

---

## 4. Phương pháp thực hiện

### 4.1. Tiền xử lý dữ liệu

- **Làm sạch:** Làm sạch dữ liệu categorical bị dính dấu nháy/khoảng trắng khi đọc bằng `np.genfromtxt` (ví dụ `"Existing Customer"`), chuẩn hoá chuỗi bằng `strip` để tránh sai lệch khi mã hoá target/feature.
- **Encoding:** Mã hoá các biến phân loại (đặc biệt là biến có thứ tự) sang dạng số để tính tương quan và phục vụ mô hình.
- **Feature selection (Correlation):** Tính tương quan Pearson giữa `Attrition_Flag` (mã hoá 0/1) và các đặc trưng (sau encode) để tham khảo/top-k.
- **Scaling (Z-score):** Chuẩn hoá đặc trưng theo train-set:
	$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

### 4.2. Cài đặt: Logistic Regression (NumPy from scratch)

Mô hình được cài đặt trong lớp `LogisticRegressionNumpy` với tối ưu bằng Gradient Descent, sử dụng phép nhân ma trận (vectorization) thay vì vòng lặp theo mẫu.

### 4.3. Công  thức toán

**(1) Tổ hợp tuyến tính**
$$z = w \cdot x + b$$

Với dữ liệu dạng ma trận $X \in \mathbb{R}^{n \times d}$:
$$z = Xw + b$$

**(2) Sigmoid**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**(3) Xác suất dự đoán**
$$\hat{p} = P(y=1\mid x) = \sigma(z)$$

**(4) Loss – Binary Cross Entropy (Log Loss)**
$$\mathcal{L}(w,b) = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\Big]$$

**(5) Gradient Descent update**

Với $\hat{p} = \sigma(Xw + b)$:
$$w \leftarrow w - \alpha\,\frac{1}{n}X^T(\hat{p} - y)$$
$$b \leftarrow b - \alpha\,\frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)$$

Trong đó $\alpha$ là learning rate.

### 4.4. Ghi chú về cài đặt

- Ưu tiên **vectorization** (ví dụ `X @ w`, `X.T @ (p - y)`) để giảm lỗi logic, tăng hiệu năng và tránh vòng lặp theo từng dòng.
- Kiểm soát shape rõ ràng bằng `reshape(-1)` hoặc thêm chiều bằng `np.newaxis` khi cần.

---

## 6. Installation & Setup

### 6.1. Requirements

- `numpy`
- `pandas` (phục vụ đọc/ghi dữ liệu nếu cần; notebook hiện tại chủ yếu dùng NumPy)
- `matplotlib`
- `seaborn`

### 6.2. Cài đặt

```bash
git clone <your-repo-url>
cd Lab2-ProDS

python -m pip install -r requirements.txt

# (khuyến nghị) nếu chưa có pandas
python -m pip install pandas
```

---

## 7. Usage

Thực hiện theo đúng thứ tự notebook:

1. Chạy `notebooks/01_data_exploration.ipynb`
	 - EDA: thống kê mô tả, phân phối biến mục tiêu, phân tích đặc trưng và tương quan.
2. Chạy `notebooks/02_preprocessing.ipynb`
	 - Tiền xử lý, encode, chọn đặc trưng tham khảo theo tương quan.
	 - Tạo và lưu các tập train/valid/test cho 2 giả thuyết (H1, H2).
3. Chạy `notebooks/03_modeling.ipynb`
	 - Huấn luyện Logistic Regression (NumPy), tuning learning rate và threshold theo **F1 trên valid**, đánh giá trên test.

---

## 8. Kết quả

### 8.1. Thiết lập đánh giá

- Do dữ liệu mất cân bằng, dự án ưu tiên **F1-score**.
- Thực hiện **hyperparameter tuning** theo F1 trên tập validation cho:
	- Learning rate (`lr`)
	- Ngưỡng phân loại (`threshold`)

### 8.2. Hypothesis Testing

- **H1 (Linear model):** sử dụng 3 biến
	- `Total_Trans_Ct`, `Total_Revolving_Bal`, `Contacts_Count_12_mon`
- **H2 (Interaction model):** sử dụng 4 biến, thêm biến tương tác
	- `Total_Trans_Ct`, `Total_Trans_Amt`, `Total_Revolving_Bal`
	- `Engagement_Score = Total_Trans_Ct * Total_Trans_Amt`

### 8.3. Metrics tổng quan

Ngưỡng tốt nhất được chọn theo valid:
- **H1:** `lr = 0.003`, `threshold = 0.37`
- **H2:** `lr = 0.3`, `threshold = 0.38`

| Dataset | H1 Accuracy | H1 Precision | H1 Recall | H1 F1 | H2 Accuracy | H2 Precision | H2 Recall | H2 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 0.8662 | 0.5931 | 0.5376 | 0.5640 | 0.8778 | 0.6256 | 0.5997 | 0.6124 |
| Valid | 0.8617 | 0.5633 | 0.5563 | 0.5597 | 0.8804 | 0.6127 | 0.6625 | 0.6366 |
| Test  | 0.8686 | 0.6014 | 0.5460 | 0.5723 | 0.8854 | 0.6497 | 0.6258 | 0.6375 |

**Kết luận ngắn:** H2 cải thiện F1 nhất quán trên Train/Valid/Test, cho thấy việc bổ sung `Total_Trans_Amt` và biến tương tác `Engagement_Score` giúp mô hình nắm bắt mức độ gắn kết khách hàng tốt hơn.

---

## 9. Project Structure

```
Lab2-ProDS/
├─ data/
│  ├─ raw/                # dữ liệu gốc (original_data.csv)
│  └─ processed/          # dữ liệu sau tiền xử lý (preprocessed_data.csv)
├─ notebooks/
│  ├─ 01_data_exploration.ipynb
│  ├─ 02_preprocessing.ipynb
│  └─ 03_modeling.ipynb
├─ src/                   # nơi đặt mã nguồn tách khỏi notebook, cho tương lai, hiện tại để trống
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## 10. Thử thách và giải pháp

- **Thử thách:** Cài đặt vectorization và xử lý lỗi broadcasting/shape trong NumPy.
	- **Solution:** Kiểm tra tính tương thích kích thước, chuẩn hoá shape bằng `np.reshape`, `reshape(-1)` hoặc thêm chiều bằng `np.newaxis` khi cần.

- **Thử thách:** Tuning learning rate và threshold thủ công (không dùng GridSearchCV).
	- **Solution:** Xây dựng vòng lặp thử nghiệm (grid) và chọn cấu hình tối ưu theo **F1-score trên validation**.

- **Thử thách:** Quá nhiều điều cần học, quá nhiều code cần đọc và xem. Không có nhiều thời gian để tối ưu code.
	- **Solution:** Ưu tiên pipeline tối thiểu nhưng đúng bản chất: làm sạch dữ liệu → chuẩn hoá → cài đặt Logistic Regression → tuning và đánh giá theo F1.

---

## 11. Future Improvements

- Thử các thuật toán khác và tự cài đặt từ đầu (ví dụ Decision Tree, Random Forest).
- Xử lý mất cân bằng lớp tốt hơn (ví dụ SMOTE) và/hoặc thử các chiến lược điều chỉnh threshold/weighting theo chi phí FN/FP.

---

## 12. Tác giả

- **Author:** Nguyễn Đồng Thanh
- **Student ID:** 23127538
- **Contact:** [Your Email/GitHub Link]

---

## 13. License

MIT License.

