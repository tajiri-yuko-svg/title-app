import io
import re
import unicodedata
import jaconv
import Levenshtein
import pandas as pd
from rapidfuzz import distance
import streamlit as st

# --- ページ基本設定 ---
st.set_page_config(page_title="高速・作品タイトル照合ツール", layout="wide")
st.title("⚡ 大容量対応 作品タイトル表記ゆれ照合ツール")
st.write("ファイルB（最大1万件）を基準に、ファイルA（最大15万件）から最適マッチを高速で抽出します。")

# --- 定数・辞書定義 ---
SYNONYM_DICT = {
    "&": "アンド",
    "＆": "アンド",
    "vs": "対",
    "ｖｓ": "対",
    "劇場版": "映画",
    "vol.": "巻",
    "vol": "巻",
}

NUM_MAP = {
    "Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "Ⅴ": "5",
    "Ⅵ": "6", "Ⅶ": "7", "Ⅷ": "8", "Ⅸ": "9", "Ⅹ": "10",
    "ⅰ": "1", "ⅱ": "2", "ⅲ": "3", "ⅳ": "4", "ⅴ": "5",
    "ⅵ": "6", "ⅶ": "7", "ⅷ": "8", "ⅸ": "9", "ⅹ": "10",
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
}

def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    return str(text).strip()

@st.cache_data
def normalize_full(text):
    """前処理（①〜③）を統合した完全正規化関数"""
    if not text:
        return ""
    # Unicode正規化 (NFKC: 全角英数・記号の半角化)
    text = unicodedata.normalize("NFKC", text)
    # 1. 括弧および括弧内文字の除去
    text = re.sub(r"[\(（【\[《〈〔].*?[\)）】\]》〉〕]", "", text)
    # 2. シノニム置換
    for key, val in SYNONYM_DICT.items():
        text = text.replace(key, val)
    # 3. シリーズ・巻数表記除去
    text = re.sub(r"(season\s*\d+|第\d+[期部巻]|章|vol\.?\d+)", "", text, flags=re.IGNORECASE)
    # 4. 数字表記の統一
    for k, v in NUM_MAP.items():
        text = text.replace(k, v)
    # 5. ひらがな・カタカナ統一
    text = jaconv.kata2hira(text)
    # 6. 小文字化
    text = text.lower()
    # 7. 記号・スペース除去
    text = re.sub(r"[^\w\u3040-\u309F\u4E00-\u9FFF]", "", text)
    return text

def extract_blocks(text, block_size=5):
    """タイトルを「前半」「中央」「後半」に分割抽出"""
    length = len(text)
    if length == 0:
        return "", "", ""
    if length <= block_size:
        return text, text, text
    head = text[:block_size]
    mid_start = max(0, (length - block_size) // 2)
    mid = text[mid_start : mid_start + block_size]
    tail = text[-block_size:]
    return head, mid, tail

def evaluate_block_matching(norm_a, norm_b, block_size=5):
    head_a, mid_a, tail_a = extract_blocks(norm_a, block_size)
    head_b, mid_b, tail_b = extract_blocks(norm_b, block_size)
    head_match = head_a == head_b and len(head_a) > 0
    mid_match = mid_a == mid_b and len(mid_a) > 0
    tail_match = tail_a == tail_b and len(tail_a) > 0
    matched_count = sum([head_match, mid_match, tail_match])
    return matched_count

# --- ファイルアップロードUI ---
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("ファイルA (file_a.xlsx / 最大15万件)", type=["xlsx", "xls"])
with col2:
    file_b = st.file_uploader("ファイルB (file_b.xlsx / 最大1万件)", type=["xlsx", "xls"])

if file_a and file_b:
    st.success("ファイルの読み込み完了。照合準備ができました。")
    if st.button("🚀 超高速照合を実行する", type="primary"):
        with st.spinner("データを読み込んでインデックスを作成中..."):
            df_a = pd.read_excel(file_a)
            df_b = pd.read_excel(file_b)

            # カラム存在チェック
            col_t_a = "タイトル" if "タイトル" in df_a.columns else df_a.columns[0]
            col_a_a = "著者" if "著者" in df_a.columns else None
            col_t_b = "タイトル" if "タイトル" in df_b.columns else df_b.columns[0]
            col_a_b = "著者" if "著者" in df_b.columns else None

            # 前処理と正規化列の事前計算（ベクトル処理）
            df_a["raw_t"] = df_a[col_t_a].apply(clean_text)
            df_a["raw_a"] = df_a[col_a_a].apply(clean_text) if col_a_a else ""
            df_a["norm_t"] = df_a["raw_t"].apply(normalize_full)

            df_b["raw_t"] = df_b[col_t_b].apply(clean_text)
            df_b["raw_a"] = df_b[col_a_b].apply(clean_text) if col_a_b else ""
            df_b["norm_t"] = df_b["raw_t"].apply(normalize_full)

            # --- 高速検索用インデックス（辞書）の作成 (ファイルAをベースに作成) ---
            # 1. 完全未加工 (タイトル+著者)
            dict_ss = {}
            # 2. 未加工タイトルのみ
            dict_s = {}
            # 3. 完全正規化タイトル
            dict_a = {}
            # 4. 先頭バケット (類似度計算用: 先頭3文字をキーにして絞り込み)
            prefix_bucket_a = {}

            for idx_a, row_a in df_a.iterrows():
                rt = row_a["raw_t"]
                ra = row_a["raw_a"]
                nt = row_a["norm_t"]
                if rt and ra:
                    dict_ss.setdefault(f"{rt}____{ra}", []).append(row_a)
                if rt:
                    dict_s.setdefault(rt, []).append(row_a)
                if nt:
                    dict_a.setdefault(nt, []).append(row_a)
                # バケット登録
                p3 = nt[:3]
                prefix_bucket_a.setdefault(p3, []).append(row_a)

            # --- 照合ループ実行 (ファイルBを基準にループ) ---
            results = []
            progress_bar = st.progress(0)
            total_b = len(df_b)

            for idx_b, row_b in df_b.iterrows():
                rt_b = row_b["raw_t"]
                ra_b = row_b["raw_a"]
                nt_b = row_b["norm_t"]

                matched = False
                best_rank, best_score, best_detail = "Rank E", 0.00, "不一致"
                best_match_row = None

                # --- Rank SS 判定 ($O(1)$) ---
                if not matched and rt_b and ra_b:
                    key_ss = f"{rt_b}____{ra_b}"
                    if key_ss in dict_ss:
                        best_match_row = dict_ss[key_ss][0]
                        best_rank, best_score, best_detail = (
                            "Rank SS",
                            1.00,
                            "未加工タイトル＆著者完全一致",
                        )
                        matched = True

                # --- Rank S 判定 ($O(1)$) ---
                if not matched and rt_b:
                    if rt_b in dict_s:
                        best_match_row = dict_s[rt_b][0]
                        best_rank, best_score, best_detail = (
                            "Rank S",
                            1.00,
                            "未加工タイトル完全一致",
                        )
                        matched = True

                # --- Rank A 判定 ($O(1)$) ---
                if not matched and nt_b:
                    if nt_b in dict_a:
                        best_match_row = dict_a[nt_b][0]
                        best_rank, best_score, best_detail = (
                            "Rank A",
                            0.90,
                            "完全正規化タイトル一致",
                        )
                        matched = True

                # --- Rank B / C / D 判定（絞り込み探索） ---
                if not matched and nt_b:
                    # 先頭3文字が一致するファイルAの候補グループのみを対象にして高速化
                    p3 = nt_b[:3]
                    candidate_list = prefix_bucket_a.get(p3, [])

                    for row_a in candidate_list:
                        nt_a = (
                            row_a.norm_t
                            if hasattr(row_a, "norm_t")
                            else row_a["norm_t"]
                        )
                        if not nt_a:
                            continue

                        # Rank B
                        block_cnt = evaluate_block_matching(nt_b, nt_a)
                        contains_7 = (nt_b in nt_a and len(nt_b) >= 7) or (
                            nt_a in nt_b and len(nt_a) >= 7
                        )
                        prefix_15 = (
                            nt_b[:15] == nt_a[:15]
                            if len(nt_b) >= 15 and len(nt_a) >= 15
                            else False
                        )

                        if block_cnt >= 2 or contains_7 or prefix_15:
                            best_rank, best_score, best_detail = (
                                "Rank B",
                                0.70,
                                "高精度部分一致/包含/先頭15文字一致",
                            )
                            best_match_row = (
                                row_a
                                if not hasattr(row_a, "_asdict")
                                else row_a._asdict()
                            )
                            matched = True
                            break

                        # 類似度計算
                        lev_score = Levenshtein.ratio(nt_b, nt_a)
                        jw_score = distance.JaroWinkler.similarity(nt_b, nt_a)
                        prefix_10 = (
                            nt_b[:10] == nt_a[:10]
                            if len(nt_b) >= 10 and len(nt_a) >= 10
                            else False
                        )

                        # Rank C
                        if (
                            lev_score >= 0.80
                            or jw_score >= 0.80
                            or prefix_10
                            or block_cnt >= 1
                        ):
                            if best_score < 0.60:
                                best_rank, best_score, best_detail = (
                                    "Rank C",
                                    0.60,
                                    f"類似度(Lev:{lev_score:.2f}) / 先頭10文字:{prefix_10}",
                                )
                                best_match_row = (
                                    row_a
                                    if not hasattr(row_a, "_asdict")
                                    else row_a._asdict()
                                )

                        # Rank D
                        elif lev_score >= 0.70 or (
                            len(nt_b) >= 7
                            and len(nt_a) >= 7
                            and nt_b[:7] == nt_a[:7]
                        ):
                            if best_score < 0.50:
                                best_rank, best_score, best_detail = (
                                    "Rank D",
                                    0.50,
                                    f"類似度(Lev:{lev_score:.2f})",
                                )
                                best_match_row = (
                                    row_a
                                    if not hasattr(row_a, "_asdict")
                                    else row_a._asdict()
                                )

                # 結果格納
                a_title = ""
                a_author = ""
                if best_match_row is not None:
                    if isinstance(best_match_row, dict):
                        a_title = best_match_row.get("raw_t", "")
                        a_author = best_match_row.get("raw_a", "")
                    elif hasattr(best_match_row, "raw_t"):
                        a_title = getattr(best_match_row, "raw_t", "")
                        a_author = getattr(best_match_row, "raw_a", "")
                    else:
                        a_title = best_match_row["raw_t"]
                        a_author = best_match_row["raw_a"]

                results.append({
                    "ファイルB_タイトル": rt_b,
                    "ファイルB_著者": ra_b,
                    "判定ランク": best_rank,
                    "スコア": best_score,
                    "マッチ詳細・条件": best_detail,
                    "ファイルA_候補タイトル": a_title if matched or best_score > 0 else "該当なし",
                    "ファイルA_候補著者": a_author if matched or best_score > 0 else "該当なし",
                })

                # プログレスバー更新 (200件ごとに更新)
                if idx_b % 200 == 0 or idx_b == total_b - 1:
                    progress_bar.progress((idx_b + 1) / total_b)

            result_df = pd.DataFrame(results)
            st.success("照合処理が完了しました！")
            st.subheader(f"📊 照合結果プレビュー (全{len(result_df)}件中 先頭100件を表示)")
            st.dataframe(result_df.head(100), use_container_width=True)

            # Excelのバイナリ生成
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False)
            processed_data = output.getvalue()

            st.download_button(
                label="📥 全件の照合結果Excelをダウンロード",
                data=processed_data,
                file_name="match_result_fast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )