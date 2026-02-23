# app.py
import streamlit as st
import pandas as pd
import math
from ortools.sat.python import cp_model

BEER_NAMES = [
    "KEN BẠC LON 330ML", "BIVINA EXPORT LON",
    "KEN BẠC LON 250ML", "KEN BẠC CHAI",
    "TIGER BẠC LON 250ML", "TIGER BẠC LON", "SÀI GÒN TRẮNG",
]
SOFT_DRINK_NAMES = ["NƯỚC NGỌT"]
MAIN_CATEGORIES  = ["GÀ", "CÁ", "LẨU", "LƯƠN", "MỰC", "ẾCH", "BỒ CÂU"]
STARTER_CATEGORY = "KHAI VỊ"
BEER_CATEGORY    = "BIA RƯỢU"
MAX_QTY          = 200
TAX_FOOD         = 1 - 0.006


def effective_price(row):
    name = str(row["Tên sản phẩm"]).upper().strip()
    cat  = str(row["Danh mục sản phẩm"]).upper().strip()
    raw  = float(row["Giá bán"])
    if cat == BEER_CATEGORY or name in [s.upper() for s in SOFT_DRINK_NAMES]:
        return int(raw)
    return round(raw * TAX_FOOD)


def solve(df, target, cfg):
    df = df.copy()
    df["Tên sản phẩm"]      = df["Tên sản phẩm"].str.upper().str.strip()
    df["Danh mục sản phẩm"] = df["Danh mục sản phẩm"].str.upper().str.strip()
    df["eff_price"] = df.apply(effective_price, axis=1)

    relevant_cats = [BEER_CATEGORY, STARTER_CATEGORY] + MAIN_CATEGORIES + [
        "THỰC ĐƠN CƠM", "CÁ", "LẨU", "LƯƠN", "MỰC", "ẾCH", "BỒ CÂU", "CHÁO", "CƠM"
    ]
    df      = df[df["Danh mục sản phẩm"].isin(relevant_cats)].reset_index(drop=True)
    n_items = len(df)
    prices  = df["eff_price"].tolist()
    target_k = target

    N_lo         = math.ceil(target / 700_000)
    N_hi         = math.floor(target / 400_000)
    N_candidates = list(range(N_lo, N_hi + 1))

    def idx_of(name):
        return df[df["Tên sản phẩm"] == name.upper()].index.tolist()

    beer_idx    = {b: idx_of(b) for b in BEER_NAMES}
    khan_idx    = idx_of("KHĂN LẠNH")
    btm_idx     = idx_of("BÁNH TRÁNG MÈ")
    soft_idx    = [
        i for i in df[df["Danh mục sản phẩm"] == BEER_CATEGORY].index.tolist()
        if df.loc[i, "Tên sản phẩm"] in [s.upper() for s in SOFT_DRINK_NAMES]
    ]
    starter_idx = [
        i for i in df[df["Danh mục sản phẩm"] == STARTER_CATEGORY].index.tolist()
        if df.loc[i, "Tên sản phẩm"] not in ["KHĂN LẠNH", "BÁNH TRÁNG MÈ", "NƯỚC SUỐI"]
    ]
    main_idx = df[df["Danh mục sản phẩm"].isin(MAIN_CATEGORIES)].index.tolist()

    best_sol   = None
    best_N     = None
    best_over = 9999

    for attempt in range(2):
        force_no_div5 = cfg["beer_no_div5"] and (attempt == 0)
        if attempt == 1:
            print("⚠️  Lượt 2: thử lại không áp dụng quy tắc chia hết 5")
        for N in N_candidates:
            model = cp_model.CpModel()
            qty   = [model.new_int_var(0, MAX_QTY, f"q_{i}") for i in range(n_items)]

            total_expr = sum(qty[i] * prices[i] for i in range(n_items))
            over = model.new_int_var(0, 3000, "over")
            model.add(total_expr == target_k + over)

            beer_totals = {}
            for bname, bidx in beer_idx.items():
                beer_totals[bname] = qty[bidx[0]] * prices[bidx[0]] if bidx else 0
            total_beer = sum(beer_totals.values())
            model.add(total_beer >= int(cfg["beer_min"] * target_k))
            model.add(total_beer <= int(cfg["beer_max"] * target_k))

            # Ken 330ml > X% tổng bia
            ken330_total = beer_totals.get("KEN BẠC LON 330ML", 0)
            if cfg["ken330_fixed_qty"] is not None:
                if beer_idx.get("KEN BẠC LON 330ML"):
                    model.add(qty[beer_idx["KEN BẠC LON 330ML"][0]] == cfg["ken330_fixed_qty"])
            else:
                if cfg["ken330_min"] is not None:
                    model.add(ken330_total * 100 > int(cfg["ken330_min"] * 100) * total_beer)

            if force_no_div5:
                for bname, bidx in beer_idx.items():
                    if not bidx:
                        continue
                    i       = bidx[0]
                    is_used = model.new_bool_var(f"used_{i}")
                    model.add(qty[i] >= 1).only_enforce_if(is_used)
                    model.add(qty[i] == 0).only_enforce_if(is_used.Not())
                    # Khi dùng: qty[i] mod 5 phải nằm trong {1,2,3,4}
                    # Tức là qty[i] mod 5 != 0
                    # Dùng: qty[i] = 5*k + r, r in [1,4]
                    k = model.new_int_var(0, 40, f"k_{i}")
                    r = model.new_int_var(1,  4, f"r_{i}")
                    model.add(qty[i] == 5 * k + r).only_enforce_if(is_used)

            if soft_idx:
                soft_total = sum(qty[i] * prices[i] for i in soft_idx)
                model.add(soft_total <= int(cfg["soft_max"] * target_k))

            if khan_idx:
                model.add(qty[khan_idx[0]] >= N)
                model.add(qty[khan_idx[0]] <= N + 2)

            if cfg["require_food"] and btm_idx:
                btm_money = qty[btm_idx[0]] * prices[btm_idx[0]]
                model.add(btm_money >= int(0.01 * target_k))
                model.add(btm_money <= int(0.02 * target_k))

            starter_used = []
            for i in starter_idx:
                b = model.new_bool_var(f"sv_{i}")
                model.add(qty[i] >= 1).only_enforce_if(b)
                model.add(qty[i] == 0).only_enforce_if(b.Not())
                starter_used.append(b)

            main_used = []
            for i in main_idx:
                b = model.new_bool_var(f"mv_{i}")
                model.add(qty[i] >= 1).only_enforce_if(b)
                model.add(qty[i] == 0).only_enforce_if(b.Not())
                main_used.append(b)

            if cfg["require_food"]:
                model.add(sum(starter_used) >= 2)
                model.add(sum(starter_used) <= 3)
                model.add(sum(main_used) >= 2)
                model.add(sum(main_used) <= 3)
            else:
                model.add(sum(starter_used) == 0)
                model.add(sum(main_used) == 0)

            allowed = (
                [bidx[0] for bidx in beer_idx.values() if bidx]
                + soft_idx + khan_idx + btm_idx + starter_idx + main_idx
            )
            for i in range(n_items):
                if i not in allowed:
                    model.add(qty[i] == 0)

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 15.0
            solver.parameters.num_search_workers  = 8
            solver.parameters.symmetry_level      = 0
            solver.parameters.linearization_level = 2
            status = solver.solve(model)

            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                sol_over = solver.value(over)
                if best_sol is None or sol_over < best_over:
                    best_sol  = [solver.value(qty[i]) for i in range(n_items)]
                    best_N    = N
                    best_over = sol_over
                break

            else:
                print(f"  attempt={attempt+1}, N={N}, {solver.status_name(status)}")

        if best_sol is not None:
            break

    if best_sol is None:
        return None

    rows = []
    for i in range(n_items):
        q = best_sol[i]
        if q == 0:
            continue
        name    = df.loc[i, "Tên sản phẩm"]
        cat     = df.loc[i, "Danh mục sản phẩm"]
        price_raw = int(df.loc[i, "Giá bán"])   # giá gốc để hiển thị
        cat = df.loc[i, "Danh mục sản phẩm"]
        name = df.loc[i, "Tên sản phẩm"]
        if cat.upper() == BEER_CATEGORY or name in [s.upper() for s in SOFT_DRINK_NAMES]:
            price = price_raw
        else:
            price = round(price_raw * TAX_FOOD)  # chỉ để hiển thị
        total_i = q * price
        tax_label = "Giữ nguyên" if (
            cat.upper() == BEER_CATEGORY or name in [s.upper() for s in SOFT_DRINK_NAMES]
        ) else "−0.6%"
        rows.append({
            "Tên món":          name.title(),
            "Phân loại":        cat.title(),
            "Số lượng":         q,
            "Đơn giá (VNĐ)":   f"{price:,}",
            "Thuế":             tax_label,
            "Thành tiền (VNĐ)": f"{total_i:,}",
            "_total_raw":       total_i,
        })

    result_df   = pd.DataFrame(rows)
    grand_total = int(result_df["_total_raw"].sum())
    result_df   = result_df.drop(columns=["_total_raw"])
    footer      = pd.DataFrame([{
        "Tên món": "TỔNG CỘNG", "Phân loại": "", "Số lượng": "",
        "Đơn giá (VNĐ)": "", "Thuế": "",
        "Thành tiền (VNĐ)": f"{grand_total:,}",
    }])
    return pd.concat([result_df, footer], ignore_index=True), grand_total, best_N


def render_invoice_table(df_items, show_tax_note=False):
    if df_items.empty:
        st.info("Không có món nào trong nhóm này.")
        return 0

    total_raw = sum(
        int(str(v).replace(",", ""))
        for v in df_items["Thành tiền (VNĐ)"]
        if str(v).replace(",", "").isdigit()
    )

    display_rows = []
    for idx, row in df_items.iterrows():
        don_gia_raw = int(str(row["Đơn giá (VNĐ)"]).replace(",", ""))
        so_luong    = int(row["Số lượng"])
        if show_tax_note:
            don_gia_goc = round(don_gia_raw / TAX_FOOD)
            thanh_tien  = so_luong * don_gia_goc
        else:
            don_gia_goc = don_gia_raw
            thanh_tien  = int(str(row["Thành tiền (VNĐ)"]).replace(",", ""))
        display_rows.append({
            "STT":          idx + 1,
            "Tên hàng hóa": row["Tên món"],
            "ĐVT":          row.get("Đơn vị tính", ""),
            "Số lượng":     so_luong,
            "Đơn giá":      f"{don_gia_goc:,}",
            "Thành tiền":   f"{thanh_tien:,}",
        })

    df_display = pd.DataFrame(display_rows)
    footer_row = pd.DataFrame([{
        "STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": None,
        "Đơn giá": "Tổng tiền thanh toán:",
        "Thành tiền": f"{total_raw:,}",
    }])
    df_display = pd.concat([df_display, footer_row], ignore_index=True)

    def style_footer(row):
        if row["Đơn giá"] == "Tổng tiền thanh toán:":
            return ["font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_display.style.apply(style_footer, axis=1),
        width="stretch",
        hide_index=True,
    )

    if show_tax_note:
        giam_tru = round(total_raw * 0.006)
        st.caption(
            f"Đã giảm **{giam_tru:,} đồng** tương ứng 20% mức tỷ lệ % để tính thuế GTGT "
            f"theo Nghị quyết số 204/2025/QH15."
        )

    return total_raw


def main():
    st.set_page_config(page_title="🍺 Tạo Hóa Đơn Nhà Hàng", page_icon="🍺", layout="wide")
    st.title("🍺 Hệ thống Tạo Hóa Đơn Nhà Hàng Tự Động")
    st.caption("Sử dụng Google OR-Tools CP-SAT Solver")

    with st.sidebar:
        st.header("⚙️ Cấu hình Hóa Đơn")
        uploaded = st.file_uploader("📂 Upload file Menu CSV", type=["csv"])
        target   = st.number_input(
            "💰 Target_Total (VNĐ)",
            min_value=100_000, max_value=50_000_000,
            value=2_151_000, step=1_000, format="%d",
        )
        st.markdown("---")
        mode = st.selectbox(
            "📋 Chế độ hóa đơn",
            ["🍺🥘 Bia + Đồ ăn (mặc định)", "🍺 Chỉ bia", "🔧 Tùy chỉnh hoàn toàn"],
        )

        if mode == "🍺 Chỉ bia":
            cfg = {
                "beer_min": 0.90, "beer_max": 0.98, "ken330_min": 0.70,
                "ken330_fixed_qty": None, "soft_max": 0.10,
                "require_food": False, "beer_no_div5": True,
            }
        elif mode == "🍺🥘 Bia + Đồ ăn (mặc định)":
            cfg = {
                "beer_min": 0.60, "beer_max": 0.89, "ken330_min": 0.80,
                "ken330_fixed_qty": None, "soft_max": 0.07,
                "require_food": True, "beer_no_div5": True,
            }
        else:
            st.markdown("**🔧 Tùy chỉnh chi tiết:**")
            beer_range = st.slider("🍺 Beer % tổng", 0.0, 1.0, (0.60, 0.89), 0.01)
            ken_mode   = st.radio("🍺 Ken Bạc 330ml", ["Theo % tổng bia", "Cố định số lượng"])
            if ken_mode == "Cố định số lượng":
                ken330_qty = st.number_input("Số lon Ken 330ml", 1, 200, 36)
                ken330_pct = None
            else:
                ken330_pct = st.slider("Ken 330ml > X% tổng bia", 0.0, 0.95, 0.80, 0.01)
                ken330_qty = None
            soft_max     = st.slider("🥤 Nước ngọt tối đa %", 0.0, 0.30, 0.07, 0.01)
            require_food = st.toggle("🥘 Bắt buộc có đồ ăn", value=True)
            beer_no_div5 = st.toggle("🔢 SL bia không chia hết 5", value=True)
            cfg = {
                "beer_min": beer_range[0], "beer_max": beer_range[1],
                "ken330_min": ken330_pct,
                "ken330_fixed_qty": ken330_qty if ken_mode == "Cố định số lượng" else None,
                "soft_max": soft_max, "require_food": require_food,
                "beer_no_div5": beer_no_div5,
            }

        st.markdown("---")
        st.markdown("""**Ràng buộc:**
- 🍺 Beer: theo % đã chọn
- 🔢 SL bia không chia hết 5 (tự động nới nếu cần)
- 🥤 Nước ngọt: theo % đã chọn
- 🧻 Khăn lạnh = N khách
- 🥘 Bánh tráng mè: 1–2% tổng
- 🥗 2–3 Khai vị, 2–3 Món chính""")

    if uploaded is None:
        st.info("👈 Vui lòng upload file Menu CSV ở thanh bên trái để bắt đầu.")
        return

    try:
        df_menu = pd.read_csv(uploaded, sep=";", encoding="utf-8-sig")
        df_menu.columns = [c.strip() for c in df_menu.columns]
        if not {"Tên sản phẩm", "Giá bán", "Danh mục sản phẩm"}.issubset(df_menu.columns):
            st.error("File CSV thiếu cột bắt buộc.")
            return
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return

    with st.expander("📋 Xem Menu đã tải lên", expanded=False):
        st.dataframe(
            df_menu[["Tên sản phẩm", "Đơn vị tính", "Giá bán", "Danh mục sản phẩm"]],
            width="stretch",
        )

    if not st.button("🚀 Tạo Hóa Đơn", type="primary", use_container_width=True):
        return

    with st.spinner("⏳ Đang chạy CP-SAT solver..."):
        result = solve(df_menu, int(target), cfg)

    if result is None:
        st.error("❌ Không tìm được nghiệm. Thử thay đổi Target_Total hoặc nới lỏng ràng buộc.")
        return

    invoice_df, grand_total, N = result
    diff = grand_total - int(target)
    st.success(f"✅ Tổng = **{grand_total:,} VNĐ** | Số khách: **{N}** | Lệch: **+{diff:,} đ**")

    rows_beer = [r for _, r in invoice_df.iterrows()
                 if r["Tên món"] != "TỔNG CỘNG" and r["Thuế"] == "Giữ nguyên"]
    rows_food = [r for _, r in invoice_df.iterrows()
                 if r["Tên món"] != "TỔNG CỘNG" and r["Thuế"] != "Giữ nguyên"]

    df_beer = pd.DataFrame(rows_beer).reset_index(drop=True)
    df_food = pd.DataFrame(rows_food).reset_index(drop=True)

    tab1, tab2 = st.tabs(["🍺 Hóa đơn Bia & Nước ngọt", "🥘 Hóa đơn Thức ăn"])

    with tab1:
        st.markdown("### 🍺 Hóa đơn Bia & Nước ngọt")
        total_beer = render_invoice_table(df_beer, show_tax_note=False)
        if not df_beer.empty:
            st.download_button(
                "⬇️ Tải hóa đơn bia (.csv)",
                data=df_beer.to_csv(index=False, encoding="utf-8-sig"),
                file_name=f"hoadon_bia_{int(target)}.csv",
                mime="text/csv",
                key="dl_beer",
            )

    with tab2:
        st.markdown("### 🥘 Hóa đơn Thức ăn")
        total_food = render_invoice_table(df_food, show_tax_note=True)
        if not df_food.empty:
            st.download_button(
                "⬇️ Tải hóa đơn thức ăn (.csv)",
                data=df_food.to_csv(index=False, encoding="utf-8-sig"),
                file_name=f"hoadon_thucao_{int(target)}.csv",
                mime="text/csv",
                key="dl_food",
            )

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🍺 Tổng bia",     f"{total_beer:,} đ", f"{total_beer/target*100:.1f}%")
    m2.metric("👥 Số khách (N)", str(N))
    m3.metric("💰 Target",       f"{target:,} đ")
    m4.metric("✅ Tổng thực tế", f"{grand_total:,} đ", delta=f"{grand_total - target:+,} đ")


if __name__ == "__main__":
    main()
