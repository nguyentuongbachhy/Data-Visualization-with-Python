# Dự án Phân tích và Trực quan hóa Dữ liệu COVID-19
## CSC10108 - Data Visualization Project

<p align="center">
  <img src="https://www.who.int/images/default-source/health-topics/coronavirus/corona-virus-getty.tmb-1920v.jpg" alt="COVID-19 Visualization" width="600"/>
</p>

## Thông tin chung

- **Tên dự án**: Phân tích và Trực quan hóa Dữ liệu COVID-19
- **Dataset**: [Novel Corona Virus 2019 Dataset](https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset)
- **Thành viên nhóm**:
  - Nguyễn Tường Bách Hỷ
  - Quách Hải Đăng
  - Lê Châu Hữu Thọ
  - Khưu Hải Châu
  - Phan Nguyễn Minh Khôi

## Mục tiêu dự án

1. Thu thập và làm sạch dữ liệu COVID-19 từ nguồn Kaggle
2. Phân tích sâu để hiểu rõ diễn biến của đại dịch COVID-19 trên toàn cầu
3. Tạo các biểu đồ trực quan nhằm khám phá và truyền tải thông tin về:
   - Diễn biến dịch bệnh theo thời gian
   - So sánh tình hình giữa các quốc gia/khu vực
   - Mối quan hệ giữa ca nhiễm, tử vong và phục hồi
   - Tác động của các yếu tố khác nhau đến tỷ lệ tử vong

## Cấu trúc thư mục

```
Project_COVID19/
├── Docs/                       # Báo cáo và tài liệu
│   └── README.md               # File này
├── Sources/                    # Jupyter notebooks
│   ├── data_collection.ipynb   # Thu thập và khám phá dữ liệu
│   ├── data_cleaning.ipynb     # Làm sạch dữ liệu
│   └── data_visualization.ipynb # Trực quan hóa dữ liệu
└── Datasets/                   # Dữ liệu
    ├── Raw/                    # Dữ liệu gốc từ Kaggle
    └── Processed/              # Dữ liệu đã xử lý
```

## Quy trình phát triển 10 ngày

### Ngày 1-2: Thu thập dữ liệu và nghiên cứu sơ bộ

#### Ngày 1: Khởi tạo dự án và thu thập dữ liệu
- **Buổi sáng**:
  - Tải dataset Novel Corona Virus 2019 từ Kaggle
  - Tạo cấu trúc thư mục dự án
  - Nghiên cứu mô tả dataset và lịch sử cập nhật
- **Buổi chiều**:
  - Tạo notebook đầu tiên (`data_collection.ipynb`)
  - Đọc và hiển thị thông tin tổng quan của các file dữ liệu
  - Phát hiện các vấn đề chất lượng dữ liệu (giá trị âm, giá trị "Unknown")
  - Lập danh sách các câu hỏi nghiên cứu tiềm năng

#### Ngày 2: Phân tích sơ bộ và hoàn thành Task 1
- **Buổi sáng**:
  - Phân tích chi tiết cấu trúc từng file dữ liệu
  - Kiểm tra số lượng hàng, cột và kiểu dữ liệu
  - Nghiên cứu code mẫu để hiểu cách tiếp cận
- **Buổi chiều**:
  - Viết phần giải thích cho Task 1 (Context, nguồn dữ liệu, phương pháp thu thập, tính hợp pháp)
  - Xác định các thách thức và cơ hội trong dữ liệu COVID-19
  - Lập kế hoạch chi tiết cho việc làm sạch dữ liệu

### Ngày 3-4: Làm sạch và tiền xử lý dữ liệu

#### Ngày 3: Bắt đầu tiền xử lý dữ liệu
- **Buổi sáng**:
  - Tạo notebook mới (`data_cleaning.ipynb`)
  - Xử lý giá trị bị thiếu trong tất cả các tệp
  - Chuẩn hóa tên quốc gia/vùng lãnh thổ
- **Buổi chiều**:
  - Xử lý giá trị âm trong dataset (sử dụng rolling average)
  - Xử lý giá trị "Unknown" trong Province/State
  - Chuyển đổi dữ liệu time series sang định dạng long format

#### Ngày 4: Hoàn thiện tiền xử lý và chuẩn bị cho trực quan hóa
- **Buổi sáng**:
  - Thêm dữ liệu dân số quốc gia để tính tỷ lệ trên đầu người
  - Tạo các nhóm quốc gia theo khu vực địa lý (châu Á, châu Âu, v.v.)
  - Xử lý outliers và dữ liệu bất thường
- **Buổi chiều**:
  - Tính toán các biến phái sinh (tỷ lệ tử vong, ca nhiễm mới hàng ngày, trung bình trượt 7 ngày)
  - Viết phần mô tả Task 2 (định nghĩa hàng, cột, kiểu dữ liệu, phân phối giá trị)
  - Lưu các DataFrame đã xử lý để sử dụng trong trực quan hóa

### Ngày 5-7: Trực quan hóa dữ liệu

#### Ngày 5: Trực quan hóa cơ bản
- **Buổi sáng**:
  - Tạo notebook mới (`data_visualization.ipynb`)
  - Thiết lập style và palette màu nhất quán
  - Tạo các biểu đồ univariate (histogram, bar chart) cho số ca nhiễm, tử vong theo quốc gia
- **Buổi chiều**:
  - Tạo các biểu đồ line chart theo thời gian cho top 10 quốc gia bị ảnh hưởng
  - Tạo các biểu đồ pie chart cho phân phối ca nhiễm/tử vong theo châu lục
  - So sánh dữ liệu gốc và dữ liệu đã làm mượt

#### Ngày 6: Trực quan hóa nâng cao
- **Buổi sáng**:
  - Tạo heatmap thể hiện tương quan giữa các biến (ca nhiễm, tử vong, phục hồi)
  - Tạo các biểu đồ scatter plot để phân tích mối quan hệ giữa dân số và số ca nhiễm
  - Tạo box plot để so sánh phân phối ca nhiễm theo khu vực
- **Buổi chiều**:
  - Tạo choropleth map thể hiện sự phân bố địa lý của đại dịch
  - Tạo animated plots để thể hiện diễn biến của dịch theo thời gian
  - Thêm các biểu đồ so sánh tỷ lệ tử vong theo khu vực địa lý

#### Ngày 7: Thực hiện phân tích sâu và trực quan hóa sáng tạo
- **Buổi sáng**:
  - Thực hiện phân tích xu hướng và dự báo đơn giản
  - Tạo biểu đồ đa trục để hiển thị nhiều biến cùng lúc
  - Xây dựng dashboard tương tác đơn giản bằng Plotly (nếu có thời gian)
- **Buổi chiều**:
  - Áp dụng K-means clustering để phân nhóm các quốc gia theo mẫu hình dịch bệnh
  - Trực quan hóa kết quả clustering
  - Viết phần phân tích và lý giải cho Task 3

### Ngày 8-9: Viết báo cáo và tổ chức Jupyter Notebooks

#### Ngày 8: Tổ chức và hoàn thiện Jupyter Notebooks
- **Buổi sáng**:
  - Rà soát và tổ chức lại code trong các notebook
  - Thêm markdown cells giải thích chi tiết từng bước
  - Đảm bảo tất cả các visualizations được hiển thị đúng với độ phân giải cao
- **Buổi chiều**:
  - Thêm tính năng xuất biểu đồ với độ phân giải cao (300 DPI)
  - Tạo các cell kết luận cho mỗi phần phân tích
  - Kiểm tra tính nhất quán giữa các notebook

#### Ngày 9: Viết báo cáo PDF
- **Buổi sáng**:
  - Tạo cấu trúc báo cáo theo yêu cầu đề bài
  - Viết phần giới thiệu, mục tiêu và phương pháp
  - Viết phần tóm tắt kết quả thu thập và làm sạch dữ liệu (Task 1 & 2)
- **Buổi chiều**:
  - Viết phần phân tích trực quan hóa dữ liệu (Task 3)
  - Chèn các biểu đồ với chất lượng cao vào báo cáo
  - Viết phần kết luận, tổng kết các phát hiện chính

### Ngày 10: Kiểm tra và hoàn thiện dự án

#### Ngày 10: Hoàn thiện và chuẩn bị nộp
- **Buổi sáng**:
  - Kiểm tra toàn bộ báo cáo và notebook
  - Đảm bảo tất cả yêu cầu đã được đáp ứng
  - Chỉnh sửa định dạng và bố cục báo cáo
- **Buổi chiều**:
  - Đóng gói tất cả các tệp theo cấu trúc yêu cầu
  - Tạo README cho datasets folder
  - Nén (zip) và chuẩn bị nộp hoặc tải lên cloud storage

## Những điểm cần lưu ý

### 1. Xử lý vấn đề chất lượng dữ liệu

Dự án này sẽ đối mặt với các vấn đề chất lượng dữ liệu quan trọng:

- **Giá trị âm** trong các cột deaths/confirmed/recovered:
  - Nguyên nhân: Hiệu chỉnh dữ liệu, phân loại lại hoặc lỗi báo cáo
  - Xử lý: Sử dụng trung bình trượt (rolling average) hoặc thay thế bằng 0 khi tính số ca mới

- **Giá trị "Unknown"** trong cột Province/State:
  - Nguyên nhân: Không xác định được vị trí cụ thể trong một quốc gia
  - Xử lý: Giữ nguyên cho phân tích cấp quốc gia, tạo danh mục riêng cho phân tích cấp tỉnh/bang

### 2. Đa dạng biểu đồ trực quan hóa

Dự án sẽ sử dụng nhiều loại biểu đồ khác nhau để trực quan hóa các khía cạnh của dữ liệu:

- **Biểu đồ theo thời gian**: Line charts, area charts, animated time series
- **Biểu đồ so sánh**: Bar charts, grouped/stacked bar charts, box plots
- **Biểu đồ phân phối**: Histograms, pie charts, violin plots
- **Biểu đồ tương quan**: Scatter plots, heatmaps, bubble charts
- **Biểu đồ địa lý**: Choropleth maps, bubble maps

### 3. Phân tích theo nhiều khía cạnh

Dự án sẽ phân tích dữ liệu COVID-19 theo nhiều góc nhìn khác nhau:

- **Phân tích theo thời gian**: Diễn biến dịch bệnh qua các giai đoạn
- **Phân tích theo không gian**: So sánh giữa các quốc gia/khu vực
- **Phân tích mối quan hệ**: Liên hệ giữa các biến số (ca nhiễm, tử vong, phục hồi)
- **Phân tích xu hướng**: Dự báo và phát hiện các điểm bùng phát
- **Phân tích phân nhóm**: Nhóm các quốc gia theo mẫu hình dịch bệnh

## Thông tin về dataset

Dự án sử dụng [Novel Corona Virus 2019 Dataset](https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset) từ Kaggle, bao gồm các file chính:

1. **`covid_19_data.csv`**: Dữ liệu tổng hợp COVID-19, bao gồm thông tin hàng ngày theo quốc gia/vùng lãnh thổ.

2. **Files time series toàn cầu**:
   - `time_series_covid_19_confirmed.csv`: Số ca nhiễm tích lũy theo thời gian
   - `time_series_covid_19_deaths.csv`: Số ca tử vong tích lũy theo thời gian
   - `time_series_covid_19_recovered.csv`: Số ca hồi phục tích lũy theo thời gian

3. **Files time series cho Hoa Kỳ** (tùy chọn sử dụng):
   - `time_series_covid_19_confirmed_US.csv`
   - `time_series_covid_19_deaths_US.csv`

## Công nghệ sử dụng

Dự án được phát triển bằng các công nghệ sau:

- **Python 3.x**
- **Thư viện xử lý dữ liệu**: Pandas, NumPy
- **Thư viện trực quan hóa**: Matplotlib, Seaborn, (tùy chọn: Plotly)
- **Môi trường phát triển**: Jupyter Notebook

## Tiến độ dự án

- [x] Ngày 1: Thu thập dữ liệu và khởi tạo dự án
- [ ] Ngày 2: Phân tích sơ bộ và hoàn thành Task 1
- [ ] Ngày 3-4: Làm sạch và tiền xử lý dữ liệu
- [ ] Ngày 5-7: Trực quan hóa dữ liệu
- [ ] Ngày 8-9: Viết báo cáo và tổ chức Jupyter Notebooks
- [ ] Ngày 10: Kiểm tra và hoàn thiện dự án

## Tham khảo

1. [Novel Corona Virus 2019 Dataset](https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset)
2. [Coronavirus COVID-19 Preprocessing & Visualization](https://www.kaggle.com/code/mhassaan1122/coronavirus-covid-19-preprocessing-visualization)
3. [WHO Coronavirus (COVID-19) Dashboard](https://covid19.who.int/)
4. Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) COVID-19 Data
5. Our World in Data COVID-19 Dataset