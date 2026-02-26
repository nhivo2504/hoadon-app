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
    df["is_kg"] = df["Đơn vị tính"].str.strip() == "Kg"
    df["solver_price"] = df["eff_price"].copy()
    df.loc[df["is_kg"], "solver_price"] = (df.loc[df["is_kg"], "eff_price"] / 10).round().astype(int)
    relevant_cats = [BEER_CATEGORY, STARTER_CATEGORY] + MAIN_CATEGORIES + [
        "THỰC ĐƠN CƠM", "CÁ", "LẨU", "LƯƠN", "MỰC", "ẾCH", "BỒ CÂU", "CHÁO", "CƠM"
    ]
    df      = df[df["Danh mục sản phẩm"].isin(relevant_cats)].reset_index(drop=True)
    n_items = len(df)
    prices   = df["solver_price"].tolist()   # ← dùng solver_price thay eff_price
    target_k = target                  # ← nhân 100 để khớp với món Kg
    N_lo         = max(1, math.floor(target / 700_000))
    N_hi         = max(N_lo, math.ceil(target / 400_000))
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
    water_idx = [
        i for i in df[df["Danh mục sản phẩm"] == STARTER_CATEGORY].index.tolist()
        if df.loc[i, "Tên sản phẩm"] == "NƯỚC SUỐI"
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
            over = model.new_int_var(0, 30_000, "over")
            model.add(total_expr == target_k + over)
            model.minimize(over)

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
                if cfg["ken330_min"] is not None and cfg["ken330_min"] > 0:
                    pass  # ken330_min constraint removed

            # [removed no_div5] if force_no_div5:
            # [removed no_div5] for bname, bidx in beer_idx.items():
            # [removed no_div5] if not bidx:
            # [removed no_div5] continue
            # [removed no_div5] i       = bidx[0]
            # [removed no_div5] is_used = model.new_bool_var(f"used_{i}")
            # [removed no_div5] model.add(qty[i] >= 1).only_enforce_if(is_used)
            # [removed no_div5] model.add(qty[i] == 0).only_enforce_if(is_used.Not())
            # [removed no_div5] # Khi dùng: qty[i] mod 5 phải nằm trong {1,2,3,4}
            # [removed no_div5] # Tức là qty[i] mod 5 != 0
            # [removed no_div5] # Dùng: qty[i] = 5*k + r, r in [1,4]
            # [removed no_div5] k = model.new_int_var(0, 40, f"k_{i}")
            # [removed no_div5] r = model.new_int_var(1,  4, f"r_{i}")
            # [removed no_div5] model.add(qty[i] == 5 * k + r).only_enforce_if(is_used)

            if soft_idx:
                soft_total = sum(qty[i] * prices[i] for i in soft_idx)
                model.add(soft_total <= int(cfg["soft_max"] * target_k))

            if khan_idx and cfg["require_food"]:
                model.add(qty[khan_idx[0]] >= N)
                model.add(qty[khan_idx[0]] <= N + 2)

            if cfg["require_food"] and btm_idx:
                btm_money = qty[btm_idx[0]] * prices[btm_idx[0]]
                if target < 1000000:
                    model.add(qty[btm_idx[0]] >= 0)
                    model.add(qty[btm_idx[0]] <= 1)
                else:
                    model.add(qty[btm_idx[0]] >= 1)
                    model.add(qty[btm_idx[0]] <= 2)

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
                if target < 1000000:
                    model.add(sum(starter_used) >= 1)
                    model.add(sum(starter_used) <= 2)
                    model.add(sum(main_used) >= 1)
                    model.add(sum(main_used) <= 2)
                else:
                    model.add(sum(starter_used) >= 2)
                    model.add(sum(starter_used) <= 3)
                    model.add(sum(main_used) >= 2)
                    model.add(sum(main_used) <= 3)
            else:
                model.add(sum(starter_used) == 0)
                model.add(sum(main_used) == 0)

            # Ràng buộc món bắt buộc
            for fname in cfg.get("forced_items", []):
                fidx = idx_of(fname)
                if not fidx:
                    continue
                fi = fidx[0]
                fname_upper = fname.upper()
                is_beer_item = any(fname_upper == b for b in BEER_NAMES)
                if is_beer_item:
                    model.add(qty[fi] >= 5)   # bia tối thiểu 5 lon
                elif df.loc[fi, "is_kg"]:
                    model.add(qty[fi] >= 8)   # kg tối thiểu 0.8kg
                else:
                    model.add(qty[fi] >= 1)

            if cfg["require_food"]:
                allowed = (
                    [bidx[0] for bidx in beer_idx.values() if bidx]
                    + soft_idx + khan_idx + btm_idx + starter_idx + main_idx + water_idx
                )
            else:
                allowed = (
                    [bidx[0] for bidx in beer_idx.values() if bidx]
                    + soft_idx
                )
            # Món Kg: tối thiểu 0.8 kg nếu được dùng
            for i in range(n_items):
                if df.loc[i, "is_kg"] and i in allowed:
                    is_used_kg = model.new_bool_var(f"kg_used_{i}")
                    model.add(qty[i] >= 1).only_enforce_if(is_used_kg)
                    model.add(qty[i] == 0).only_enforce_if(is_used_kg.Not())
                    model.add(qty[i] >= 8).only_enforce_if(is_used_kg)  # 0.8 kg min

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
        q_raw = best_sol[i]
        if q_raw == 0:
            continue
        name      = df.loc[i, "Tên sản phẩm"]
        cat       = df.loc[i, "Danh mục sản phẩm"]
        is_kg     = df.loc[i, "is_kg"]
        price_raw = int(df.loc[i, "Giá bán"])

        # ← THÊM MỚI: nếu món Kg thì qty thực = q_raw / 100
        q_display = round(q_raw / 10, 1) if is_kg else q_raw

        if cat.upper() == BEER_CATEGORY or name in [s.upper() for s in SOFT_DRINK_NAMES]:
            price = price_raw
        else:
            price = round(price_raw * TAX_FOOD)

        total_i = round(q_display * price)  # ← dùng q_display để tính tiền

        tax_label = "Giữ nguyên" if (
            cat.upper() == BEER_CATEGORY or name in [s.upper() for s in SOFT_DRINK_NAMES]
        ) else "−0.6%"

        rows.append({
            "Tên món":          name.title(),
            "Phân loại":        cat.title(),
            "Đơn vị tính":      str(df.loc[i, "Đơn vị tính"]) if "Đơn vị tính" in df.columns else "",
            "Số lượng":         q_display,   # ← hiển thị 1.50 thay vì 150
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
    for stt, (_, row) in enumerate(df_items.iterrows(), start=1):
        don_gia_str = str(row["Đơn giá (VNĐ)"]).replace(",", "")
        if not don_gia_str.isdigit():
            continue
        don_gia_raw = int(don_gia_str)
        so_luong    = row["Số lượng"]
        dvt         = row["Đơn vị tính"] if "Đơn vị tính" in row.index else ""
        thanh_tien_goc = int(str(row["Thành tiền (VNĐ)"]).replace(",", ""))
        if show_tax_note:
            don_gia_hien   = round(don_gia_raw / TAX_FOOD)
            thanh_tien_hien = round(float(str(so_luong)) * don_gia_hien)
        else:
            don_gia_hien    = don_gia_raw
            thanh_tien_hien = thanh_tien_goc
        display_rows.append({
            "STT":          str(stt),
            "Tên hàng hóa": row["Tên món"],
            "ĐVT":          str(dvt),
            "Số lượng":     str(so_luong),
            "Đơn giá":      f"{don_gia_hien:,}",
            "Thành tiền":   f"{thanh_tien_hien:,}",
        })

    df_display = pd.DataFrame(display_rows)

    if show_tax_note:
        thanh_tien_truoc = sum(
            int(str(r["Thành tiền"]).replace(",", ""))
            for r in display_rows
        )
        giam_tru    = round(thanh_tien_truoc * 0.006)
        tong_tt     = thanh_tien_truoc - giam_tru
        footer_rows = pd.DataFrame([
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Thành tiền:",          "Thành tiền": f"{thanh_tien_truoc:,}"},
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Thuế giảm trừ (0.6%):", "Thành tiền": f"-{giam_tru:,}"},
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Tổng tiền thanh toán:", "Thành tiền": f"{tong_tt:,}"},
        ])
        return_total = tong_tt
    else:
        tong_tt  = sum(
            int(str(r["Thành tiền"]).replace(",", ""))
            for r in display_rows
        )
        footer_rows = pd.DataFrame([
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Tổng tiền thanh toán:", "Thành tiền": f"{tong_tt:,}"},
        ])
        return_total = tong_tt

    df_display = pd.concat([df_display, footer_rows], ignore_index=True)

    def style_footer(row):
        if row["Đơn giá"] in ("Tổng tiền thanh toán:", "Thành tiền:", "Thuế giảm trừ (0.6%):"):
            return ["font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_display.style.apply(style_footer, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    if show_tax_note:
        st.caption(
            f"Đã giảm **{giam_tru:,} đồng** tương ứng 20% mức tỷ lệ % để tính thuế GTGT "
            f"theo Nghị quyết số 204/2025/QH15."
        )

    return return_total



def df_to_editable(df_rows: "pd.DataFrame") -> "pd.DataFrame":
    rows = []
    for _, row in df_rows.iterrows():
        if row["Tên món"] == "TỔNG CỘNG":
            continue
        don_gia_str = str(row["Đơn giá (VNĐ)"]).replace(",", "")
        if not don_gia_str.isdigit():
            continue
        rows.append({
            "Tên món": row["Tên món"],
            "Đơn vị tính": row.get("Đơn vị tính", ""),
            "Số lượng": float(row["Số lượng"]),
            "don_gia_eff": int(don_gia_str),
            "is_food": row["Thuế"] != "Giữ nguyên",
        })
    return pd.DataFrame(rows)


def on_edit_change(key: str):
    edited = st.session_state[key]
    df_key = key.replace("_editor", "")
    df = st.session_state[df_key].copy()
    for idx_str, changes in edited.get("edited_rows", {}).items():
        idx = int(idx_str)
        for col, val in changes.items():
            df.at[idx, col] = val
    st.session_state[df_key] = df


def recalc_and_render(edit_df: "pd.DataFrame", is_food: bool) -> int:
    if edit_df.empty:
        st.info("Không có món nào.")
        return 0
    display_rows = []
    for stt, row in edit_df.iterrows():
        sl = float(row["Số lượng"])
        don_gia_eff = int(row["don_gia_eff"])
        is_kg = str(row["Đơn vị tính"]).strip() == "Kg"
        sl_str = f"{sl:.1f}" if is_kg else str(int(sl))
        if is_food:
            don_gia_hien = round(don_gia_eff / TAX_FOOD)
            thanh_tien_hien = round(sl * don_gia_hien)
        else:
            don_gia_hien = don_gia_eff
            thanh_tien_hien = round(sl * don_gia_hien)
        display_rows.append({
            "STT": str(stt + 1),
            "Tên hàng hóa": row["Tên món"],
            "ĐVT": row["Đơn vị tính"],
            "Số lượng": sl_str,
            "Đơn giá": f"{don_gia_hien:,}",
            "Thành tiền": f"{thanh_tien_hien:,}",
            "_tt_raw": thanh_tien_hien,
        })
    thanh_tien_truoc = sum(r["_tt_raw"] for r in display_rows)
    if is_food:
        giam_tru = round(thanh_tien_truoc * 0.006)
        tong_tt = thanh_tien_truoc - giam_tru
        footer_rows = [
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Thành tiền:", "Thành tiền": f"{thanh_tien_truoc:,}", "_tt_raw": 0},
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Thuế giảm trừ (0.6%):", "Thành tiền": f"-{giam_tru:,}", "_tt_raw": 0},
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Tổng tiền thanh toán:", "Thành tiền": f"{tong_tt:,}", "_tt_raw": 0},
        ]
        st.caption(f"Đã giảm **{giam_tru:,} đồng** theo NQ 204/2025/QH15.")
    else:
        tong_tt = thanh_tien_truoc
        footer_rows = [
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Tổng tiền thanh toán:", "Thành tiền": f"{tong_tt:,}", "_tt_raw": 0},
        ]
    df_show = pd.DataFrame([{k: v for k, v in r.items() if k != "_tt_raw"}
                             for r in display_rows] + footer_rows)
    def style_footer(row):
        if row["Đơn giá"] in ("Tổng tiền thanh toán:", "Thành tiền:", "Thuế giảm trừ (0.6%):"):
            return ["font-weight: bold"] * len(row)
        return [""] * len(row)
    st.dataframe(df_show.style.apply(style_footer, axis=1),
                 use_container_width=True, hide_index=True)
    return tong_tt


def df_to_editable(df_rows: "pd.DataFrame") -> "pd.DataFrame":
    rows = []
    for _, row in df_rows.iterrows():
        if row["Tên món"] == "TỔNG CỘNG":
            continue
        don_gia_str = str(row["Đơn giá (VNĐ)"]).replace(",", "")
        if not don_gia_str.isdigit():
            continue
        rows.append({
            "Tên món": row["Tên món"],
            "Đơn vị tính": row.get("Đơn vị tính", ""),
            "Số lượng": float(row["Số lượng"]),
            "don_gia_eff": int(don_gia_str),
            "is_food": row["Thuế"] != "Giữ nguyên",
        })
    return pd.DataFrame(rows)


def on_edit_change(key: str):
    edited = st.session_state[key]
    df_key = key.replace("_editor", "")
    df = st.session_state[df_key].copy()
    for idx_str, changes in edited.get("edited_rows", {}).items():
        idx = int(idx_str)
        for col, val in changes.items():
            df.at[idx, col] = val
    st.session_state[df_key] = df


def recalc_and_render(edit_df: "pd.DataFrame", is_food: bool) -> int:
    if edit_df.empty:
        st.info("Không có món nào.")
        return 0
    display_rows = []
    for stt, row in edit_df.iterrows():
        sl = float(row["Số lượng"])
        don_gia_eff = int(row["don_gia_eff"])
        is_kg = str(row["Đơn vị tính"]).strip() == "Kg"
        sl_str = f"{sl:.1f}" if is_kg else str(int(sl))
        if is_food:
            don_gia_hien = round(don_gia_eff / TAX_FOOD)
            thanh_tien_hien = round(sl * don_gia_hien)
        else:
            don_gia_hien = don_gia_eff
            thanh_tien_hien = round(sl * don_gia_hien)
        display_rows.append({
            "STT": str(stt + 1),
            "Tên hàng hóa": row["Tên món"],
            "ĐVT": row["Đơn vị tính"],
            "Số lượng": sl_str,
            "Đơn giá": f"{don_gia_hien:,}",
            "Thành tiền": f"{thanh_tien_hien:,}",
            "_tt_raw": thanh_tien_hien,
        })
    thanh_tien_truoc = sum(r["_tt_raw"] for r in display_rows)
    if is_food:
        giam_tru = round(thanh_tien_truoc * 0.006)
        tong_tt = thanh_tien_truoc - giam_tru
        footer_rows = [
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Thành tiền:", "Thành tiền": f"{thanh_tien_truoc:,}", "_tt_raw": 0},
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Thuế giảm trừ (0.6%):", "Thành tiền": f"-{giam_tru:,}", "_tt_raw": 0},
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Tổng tiền thanh toán:", "Thành tiền": f"{tong_tt:,}", "_tt_raw": 0},
        ]
        st.caption(f"Đã giảm **{giam_tru:,} đồng** theo NQ 204/2025/QH15.")
    else:
        tong_tt = thanh_tien_truoc
        footer_rows = [
            {"STT": "", "Tên hàng hóa": "", "ĐVT": "", "Số lượng": "",
             "Đơn giá": "Tổng tiền thanh toán:", "Thành tiền": f"{tong_tt:,}", "_tt_raw": 0},
        ]
    df_show = pd.DataFrame([{k: v for k, v in r.items() if k != "_tt_raw"}
                             for r in display_rows] + footer_rows)
    def style_footer(row):
        if row["Đơn giá"] in ("Tổng tiền thanh toán:", "Thành tiền:", "Thuế giảm trừ (0.6%):"):
            return ["font-weight: bold"] * len(row)
        return [""] * len(row)
    st.dataframe(df_show.style.apply(style_footer, axis=1),
                 use_container_width=True, hide_index=True)
    return tong_tt

def main():
    st.set_page_config(page_title="🍺 Tạo Hóa Đơn Nhà Hàng", page_icon="🍺", layout="wide")
    st.title("🍺 Hệ thống Tạo Hóa Đơn Nhà Hàng Tự Động")
    st.caption("Sử dụng Google OR-Tools CP-SAT Solver")

    with st.sidebar:
        st.header("⚙️ Cấu hình Hóa Đơn")
        uploaded = st.file_uploader("📂 Upload file Menu CSV", type=["csv"])
        st.markdown("### 💰 Nhập Giá Trị Hóa Đơn Mong Muốn (VNĐ)")
        target = st.number_input(
            "",
            min_value=100_000, max_value=50_000_000,
            value=None, step=1_000, format="%d",
            placeholder="Ví dụ: 2,151,000",
        )
        if target is None:
            target = 2_151_000

        if target < 700_000:
            st.warning("⚠️ Target dưới 700,000đ — tự động chuyển sang **Chỉ bia**.")
            auto_beer_only = True
        else:
            auto_beer_only = False

        st.markdown("---")
        st.markdown("**🍽️ Món bắt buộc:**")
        MON_LIST = [
            "", "BIVINA EXPORT LON", "KEN BẠC LON 250ML", "KEN BẠC LON 330ML",
            "KEN BẠC CHAI", "TIGER BẠC LON", "TIGER BẠC LON 250ML", "SÀI GÒN TRẮNG",
            "NƯỚC NGỌT", "NƯỚC SUỐI", "GÀ TA 2 MÓN", "GÀ KHO SẢ GỪNG", "KHOAI TÂY CHIÊN", "CÁ CHÉP NẤU RIÊU",
        ]
        mon_bb_1 = st.selectbox("Món bắt buộc 1", MON_LIST, index=0)
        mon_bb_2 = st.selectbox("Món bắt buộc 2", MON_LIST, index=0)
        forced_items = [m for m in [mon_bb_1, mon_bb_2] if m.strip()]

        st.markdown("---")
        mode_options = ["🍺🥘 Bia + Đồ ăn (mặc định)", "🍺 Chỉ bia", "🔧 Tùy chỉnh hoàn toàn"]
        mode = st.selectbox(
            "📋 Chế độ hóa đơn",
            mode_options,
            index=1 if auto_beer_only else 0,
            disabled=auto_beer_only,
        )

        if auto_beer_only or mode == "🍺 Chỉ bia":
            cfg = {
                "beer_min": 0.90, "beer_max": 0.98, "ken330_min": None,
                "ken330_fixed_qty": None, "soft_max": 0.10,
                "require_food": False, "beer_no_div5": False,
                "forced_items": forced_items,
            }
        elif mode == "🍺🥘 Bia + Đồ ăn (mặc định)":
            cfg = {
                "beer_min": 0.60, "beer_max": 0.89, "ken330_min": None,
                "ken330_fixed_qty": None, "soft_max": 0.07,
                "require_food": True, "beer_no_div5": False,
                "forced_items": forced_items,
            }
        else:
            st.markdown("**🔧 Tùy chỉnh chi tiết:**")
            beer_range = st.slider("🍺 Beer % tổng", 0.0, 1.0, (0.60, 0.89), 0.01)

            soft_max     = st.slider("🥤 Nước ngọt tối đa %", 0.0, 0.30, 0.07, 0.01)
            require_food = st.toggle("🥘 Bắt buộc có đồ ăn", value=True)

            cfg = {
                "beer_min": beer_range[0], "beer_max": beer_range[1],
                "ken330_min": None,
                "ken330_fixed_qty": None,
                "soft_max": soft_max, "require_food": require_food,
                "beer_no_div5": False,
                "forced_items": forced_items,
            }

        # Override tùy chỉnh nếu target < 700k
        if auto_beer_only:
            cfg["require_food"] = False
            cfg["beer_min"]     = 0.90
            cfg["beer_max"]     = 0.98

        st.markdown("---")
        st.markdown("""**Ràng buộc:**
- 🍺 Beer: theo % đã chọn
- 🥤 Nước ngọt: theo % đã chọn
- 🧻 Khăn lạnh = N khách
- 🥘 Bánh tráng mè: 0–2 cái (tuỳ bill)
- 🥗 1–3 Khai vị, 1–3 Món chính""")

    import os
    if uploaded is not None:
        try:
            df_menu = pd.read_csv(uploaded, sep=";", encoding="utf-8-sig")
            df_menu.columns = [c.strip() for c in df_menu.columns]
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")
            return
    else:
        default_path = os.path.join(os.path.dirname(__file__), "menu_default.csv")
        if os.path.exists(default_path):
            df_menu = pd.read_csv(default_path, sep=";", encoding="utf-8-sig")
            df_menu.columns = [c.strip() for c in df_menu.columns]
            name_file = os.path.join(os.path.dirname(__file__), "menu_default_name.txt")
            if os.path.exists(name_file):
                with open(name_file, "r", encoding="utf-8") as nf:
                    menu_filename = nf.read().strip()
            else:
                menu_filename = "menu_default.csv"
            st.info(f"📋 Đang dùng menu mặc định: **{menu_filename}**. Upload CSV mới để thay đổi.")
        else:
            st.info("👈 Vui lòng upload file Menu CSV ở thanh bên trái để bắt đầu.")
            return
    if not {"Tên sản phẩm", "Giá bán", "Danh mục sản phẩm"}.issubset(df_menu.columns):
        st.error("File CSV thiếu cột bắt buộc.")
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

    # Lưu bản gốc solver vào session_state
    st.session_state["invoice_df_orig"] = invoice_df.copy()
    _rows_beer_init = invoice_df[
        (invoice_df["Tên món"] != "TỔNG CỘNG") & (invoice_df["Thuế"] == "Giữ nguyên")
    ].reset_index(drop=True)
    _rows_food_init = invoice_df[
        (invoice_df["Tên món"] != "TỔNG CỘNG") & (invoice_df["Thuế"] != "Giữ nguyên")
    ].reset_index(drop=True)
    st.session_state["edit_beer"] = df_to_editable(_rows_beer_init)
    st.session_state["edit_food"] = df_to_editable(_rows_food_init)

    # Lưu bản gốc solver vào session_state
    st.session_state["invoice_df_orig"] = invoice_df.copy()
    _rows_beer_init = invoice_df[
        (invoice_df["Tên món"] != "TỔNG CỘNG") & (invoice_df["Thuế"] == "Giữ nguyên")
    ].reset_index(drop=True)
    _rows_food_init = invoice_df[
        (invoice_df["Tên món"] != "TỔNG CỘNG") & (invoice_df["Thuế"] != "Giữ nguyên")
    ].reset_index(drop=True)
    st.session_state["edit_beer"] = df_to_editable(_rows_beer_init)
    st.session_state["edit_food"] = df_to_editable(_rows_food_init)
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
