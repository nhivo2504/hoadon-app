# 🍺 Hệ thống Tạo Hóa Đơn Nhà Hàng Tự Động

App Streamlit tự động tạo hóa đơn dùng **Google OR-Tools CP-SAT Solver**.

- **Deploy:** https://quangio-hoadon-1pos.streamlit.app/
- **Repo:** https://github.com/nhivo2504/hoadon-app
- **Local:** `/Users/keira/Library/CloudStorage/OneDrive-Personal/Quan Gio/hoadon-app/`

## Stack & File quan trọng
- `app.py` — toàn bộ logic
- `menu_default.csv` — menu mặc định (tự load)
- `menu_default_name.txt` — tên hiển thị menu gốc
- `.venv/` — Python 3.14

## Logic nghiệp vụ
- `TAX_FOOD = 1 - 0.006` — đồ ăn giảm 0.6% theo NQ 204/2025/QH15
- Bia & nước ngọt: giữ nguyên giá gốc
- Món ĐVT = `"Kg"`: solver qty ×10, hiển thị 1 chữ số thập phân
- `target_k = target` (không nhân 100)
- `solver_price` món Kg = `eff_price / 10`

## Ràng buộc mặc định (Bia + Đồ ăn)
- Bia: 60–89% tổng | Ken 330ml > 80% tổng bia
- Nước ngọt ≤ 7% | Khăn lạnh = N khách
- Bánh tráng mè: 1–2% | 2–3 khai vị | 2–3 món chính
- SL bia không chia hết 5

## Cấu trúc bảng Hóa đơn Thức ăn
STT | Tên hàng hóa | ĐVT | Số lượng | Đơn giá (gốc) | Thành tiền
...
Thành tiền: X,XXX,XXX
Thuế giảm trừ (0.6%): -XX,XXX
Tổng tiền thanh toán: X,XXX,XXX

## Sidebar UI
1. Upload menu CSV (fallback: `menu_default.csv`)
2. Nhập Giá Trị Hóa Đơn Mong Muốn — placeholder mờ, default 2,151,000
3. Món bắt buộc 1 & 2 — dropdown
4. Chế độ: Bia+Đồ ăn / Chỉ bia / Tùy chỉnh

## Lệnh hay dùng
```bash
cd "/Users/keira/Library/CloudStorage/OneDrive-Personal/Quan Gio/hoadon-app"
source .venv/bin/activate
streamlit run app.py

git add app.py && git commit -m "mô tả" && git push origin main

# Cập nhật menu mới
cp "/path/Menu_moi.csv" menu_default.csv
echo "Menu_moi.csv" > menu_default_name.txt
git add menu_default.csv menu_default_name.txt && git commit -m "update menu" && git push
