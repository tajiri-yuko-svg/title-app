import io
import re
import jaconv
import Levenshtein
import pandas as pd
from rapidfuzz import distance
import streamlit as st

# --- ページ設定 ---
st.set_page_config(
    page_title="作品タイトル表記ゆれ照合アプリ", layout="wide"
)

st.title("📚 作品タイトル表記ゆれ照合ツール")
st.write(
    "2つのExcelファイルをアップロードして照合を実行すると、表記ゆれを考慮した高精度な結果が出力されます。"
)

# --- 照合ロジック（辞書・関数） ---
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
    "Ⅰ": "1",
    "Ⅱ": "2",
    "Ⅲ": "3",
    "Ⅳ": "4",
    "Ⅴ": "5",
    "Ⅵ": "6",
    "Ⅶ": "7",
    "Ⅷ": "8",
    "Ⅸ": "9",
    "Ⅹ": "10",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
}


def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    return str(text).strip()


def normalize_full(text):
    if not text:
        return ""
    text = re.sub(r"[\(（【\[《〈〔].*?[\)）】\]》〉〕]", "", text)
    for key, val in SYNONYM_DICT.items():
        text = text.replace(key, val)
    text = re.sub(
        r"(season\s*\d+|第\d+[期部巻]|章|vol\.?\d+)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    for k, v in NUM_MAP.items():
        text = text.replace(k, v)
    text = jaconv.zen2han(text, kana=False, digit=True, ascii=True)
    text = jaconv.h2z(text, kana=True, ascii=False, digit=False)
    text = jaconv.kata2hira(text)
    text = text.lower()
    text = re.sub(r"[^\w\u3040-\u309F\u4E00-\u9FFF]", "", text)
    return text


def extract_blocks(text, block_size=5):
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
    details = []
    if head_match:
        details.append("前半一致")
    if mid_match:
        details.append("中央一致")
    if tail_match:
        details.append("後半一致")
    detail_str = (
        f"分割一致({matched_count}/3: " + ",".join(details) + ")"
        if details
        else "分割一致なし"
    )
    return matched_count, detail_str


def calc_similarity_scores(s1, s2):
    if not s1 or not s2:
        return 0.0, 0.0
    lev_score = Levenshtein.ratio(s1, s2)
    jw_score = distance.JaroWinkler.similarity(s1, s2)
    return lev_score, jw_score


def evaluate_match(title_a, author_a, title_b, author_b):
    raw_t_a, raw_a_a = clean_text(title_a), clean_text(author_a)
    raw_t_b, raw_a_b = clean_text(title_b), clean_text(author_b)

    if raw_t_a and raw_t_a == raw_t_b and raw_a_a and raw_a_a == raw_a_b:
        return "Rank SS", 1.00, "未加工タイトル＆著者完全一致"
    if raw_t_a and raw_t_a == raw_t_b:
        return "Rank S", 1.00, "未加工タイトル完全一致"

    norm_t_a, norm_t_b = normalize_full(raw_t_a), normalize_full(raw_t_b)
    if not norm_t_a or not norm_t_b:
        return "Rank E", 0.00, "不一致（文字なし）"

    if norm_t_a == norm_t_b:
        return "Rank A", 0.90, "完全正規化タイトル一致"

    block_matched_count, block_detail = evaluate_block_matching(
        norm_t_a, norm_t_b, block_size=5
    )
    if block_matched_count >= 2:
        return "Rank B", 0.75, f"高精度部分一致 [{block_detail}]"

    contains_7chars = (
        norm_t_a in norm_t_b and len(norm_t_a) >= 7
    ) or (norm_t_b in norm_t_a and len(norm_t_b) >= 7)
    prefix_15 = (
        norm_t_a[:15] == norm_t_b[:15]
        if len(norm_t_a) >= 15 and len(norm_t_b) >= 15
        else False
    )

    if contains_7chars or prefix_15:
        match_detail = (
            "一方が他方を含む(7文字以上)"
            if contains_7chars
            else "先頭15文字一致"
        )
        return "Rank B", 0.70, match_detail

    lev_score, jw_score = calc_similarity_scores(norm_t_a, norm_t_b)
    prefix_10 = (
        norm_t_a[:10] == norm_t_b[:10]
        if len(norm_t_a) >= 10 and len(norm_t_b) >= 10
        else False
    )

    if (
        lev_score >= 0.80
        or jw_score >= 0.80
        or prefix_10
        or block_matched_count >= 1
    ):
        extra_info = f" [{block_detail}]" if block_matched_count == 1 else ""
        return (
            "Rank C",
            0.60,
            f"類似度(Lev:{lev_score:.2f}, JW:{jw_score:.2f}){extra_info}",
        )

    prefix_7 = (
        norm_t_a[:7] == norm_t_b[:7]
        if len(norm_t_a) >= 7 and len(norm_t_b) >= 7
        else False
    )
    if lev_score >= 0.70 or prefix_7:
        return (
            "Rank D",
            0.50,
            f"類似度(Lev:{lev_score:.2f}) / 先頭7文字:{prefix_7}",
        )

    return "Rank E", 0.00, "不一致"


# --- 画面レイアウト（ファイルアップロード） ---
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader(
        "ファイルA（基幹データ）をアップロード", type=["xlsx", "xls"]
    )
with col2:
    file_b = st.file_uploader(
        "ファイルB（比較データ）をアップロード", type=["xlsx", "xls"]
    )

if file_a and file_b:
    df_a = pd.read_excel(file_a)
    df_b = pd.read_excel(file_b)

    st.success("ファイルの読み込みに成功しました。")

    if st.button("🚀 照合を実行する", type="primary"):
        progress_bar = st.progress(0)
        results = []
        total_rows = len(df_a)

        rank_order = {
            "Rank SS": 6,
            "Rank S": 5,
            "Rank A": 4,
            "Rank B": 3,
            "Rank C": 2,
            "Rank D": 1,
            "Rank E": 0,
        }

        for idx_a, row_a in df_a.iterrows():
            title_a = row_a.get("タイトル", "")
            author_a = row_a.get("著者", "")

            best_rank, best_score, best_detail = "Rank E", 0.00, ""
            best_match_row = None

            for idx_b, row_b in df_b.iterrows():
                title_b = row_b.get("タイトル", "")
                author_b = row_b.get("著者", "")

                rank, score, detail = evaluate_match(
                    title_a, author_a, title_b, author_b
                )

                if (rank_order[rank] > rank_order[best_rank]) or (
                    rank_order[rank] == rank_order[best_rank]
                    and score > best_score
                ):
                    best_rank, best_score, best_detail = rank, score, detail
                    best_match_row = row_b
                    if best_rank == "Rank SS":
                        break

            results.append(
                {
                    "ファイルA_タイトル": title_a,
                    "ファイルA_著者": author_a,
                    "判定ランク": best_rank,
                    "スコア": best_score,
                    "マッチ詳細・条件": best_detail,
                    "ファイルB_候補タイトル": (
                        best_match_row.get("タイトル", "")
                        if best_match_row is not None and best_rank != "Rank E"
                        else "該当なし"
                    ),
                    "ファイルB_候補著者": (
                        best_match_row.get("著者", "")
                        if best_match_row is not None and best_rank != "Rank E"
                        else "該当なし"
                    ),
                }
            )

            # プログレスバー更新
            progress_bar.progress((idx_a + 1) / total_rows)

        result_df = pd.DataFrame(results)

        st.subheader("📊 照合結果プレビュー")
        st.dataframe(result_df, use_container_width=True)

        # Excelダウンロードボタンの準備
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False)
        processed_data = output.getvalue()

        st.download_button(
            label="📥 照合結果Excelをダウンロード",
            data=processed_data,
            file_name="match_result_streamlit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )