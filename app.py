import os
import re
import jaconv
import Levenshtein
import pandas as pd
from rapidfuzz import distance

# --- シノニム（同義語）辞書 ---
SYNONYM_DICT = {
    "&": "アンド",
    "＆": "アンド",
    "vs": "対",
    "ｖｓ": "対",
    "劇場版": "映画",
    "vol.": "巻",
    "vol": "巻",
}

# --- ローマ数字・漢数字変換マップ ---
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
    "ⅰ": "1",
    "ⅱ": "2",
    "ⅲ": "3",
    "ⅳ": "4",
    "ⅴ": "5",
    "ⅵ": "6",
    "ⅶ": "7",
    "ⅷ": "8",
    "ⅸ": "9",
    "ⅹ": "10",
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
    """【前処理】完全正規化関数"""
    if not text:
        return ""

    # 1. 括弧および括弧内文字の除去
    text = re.sub(r"[\(（【\[《〈〔].*?[\)）】\]》〉〕]", "", text)

    # 2. シノニム置き換え
    for key, val in SYNONYM_DICT.items():
        text = text.replace(key, val)

    # 3. シリーズ・巻数・期数表記の除去
    text = re.sub(
        r"(season\s*\d+|第\d+[期部巻]|章|vol\.?\d+)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 4. 数字表記の統一
    for k, v in NUM_MAP.items():
        text = text.replace(k, v)

    # 5. 英数カナ全半角統一
    text = jaconv.zen2han(text, kana=False, digit=True, ascii=True)

    # 6. かな統一（ひらがな化）
    text = jaconv.h2z(text, kana=True, ascii=False, digit=False)
    text = jaconv.kata2hira(text)

    # 7. 英小文字化
    text = text.lower()

    # 8. 記号・スペース除去
    text = re.sub(r"[^\w\u3040-\u309F\u4E00-\u9FFF]", "", text)

    return text


def extract_blocks(text, block_size=5):
    """タイトルを「前半」「中央」「後半」の3ブロック(各5文字)に分割抽出する関数"""
    length = len(text)
    if length == 0:
        return "", "", ""

    # 5文字未満の場合はそのまま渡す
    if length <= block_size:
        return text, text, text

    # 前半5文字
    head = text[:block_size]

    # 中央5文字
    mid_start = max(0, (length - block_size) // 2)
    mid = text[mid_start : mid_start + block_size]

    # 後半5文字
    tail = text[-block_size:]

    return head, mid, tail


def evaluate_block_matching(norm_a, norm_b, block_size=5):
    """前半・中央・後半のブロック照合を行い、一致数をカウントする"""
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

    detail_str = f"分割一致({matched_count}/3: " + ",".join(details) + ")" if details else "分割一致なし"
    return matched_count, detail_str


def calc_similarity_scores(s1, s2):
    if not s1 or not s2:
        return 0.0, 0.0
    lev_score = Levenshtein.ratio(s1, s2)
    jw_score = distance.JaroWinkler.similarity(s1, s2)
    return lev_score, jw_score


def evaluate_match(title_a, author_a, title_b, author_b):
    raw_t_a = clean_text(title_a)
    raw_a_a = clean_text(author_a)
    raw_t_b = clean_text(title_b)
    raw_a_b = clean_text(author_b)

    # --- Rank SS: 未加工タイトル＆著者完全一致 ---
    if raw_t_a and raw_t_a == raw_t_b and raw_a_a and raw_a_a == raw_a_b:
        return "Rank SS", 1.00, "未加工タイトル＆著者完全一致"

    # --- Rank S: 未加工タイトル完全一致 ---
    if raw_t_a and raw_t_a == raw_t_b:
        return "Rank S", 1.00, "未加工タイトル完全一致"

    norm_t_a = normalize_full(raw_t_a)
    norm_t_b = normalize_full(raw_t_b)

    if not norm_t_a or not norm_t_b:
        return "Rank E", 0.00, "不一致（文字なし）"

    # --- Rank A: 完全正規化タイトル一致 ---
    if norm_t_a == norm_t_b:
        return "Rank A", 0.90, "完全正規化タイトル一致"

    # --- 【新規】分割ブロック照合（前半・中央・後半 5文字） ---
    block_matched_count, block_detail = evaluate_block_matching(
        norm_t_a, norm_t_b, block_size=5
    )

    # 分割照合で2ブロック以上（例: 前半＋中央、あるいは前半＋後半）一致した場合は Rank B 以上にする
    if block_matched_count >= 2:
        return "Rank B", 0.75, f"高精度部分一致 [{block_detail}]"

    # --- Rank B (従来条件): 包含関係(7文字以上) OR 先頭15文字一致 ---
    contains_7chars = (
        (norm_t_a in norm_t_b and len(norm_t_a) >= 7)
        or (norm_t_b in norm_t_a and len(norm_t_b) >= 7)
    )
    prefix_15 = (
        norm_t_a[:15] == norm_t_b[:15]
        if len(norm_t_a) >= 15 and len(norm_t_b) >= 15
        else False
    )

    if contains_7chars or prefix_15:
        match_detail = "一方が他方を含む(7文字以上)" if contains_7chars else "先頭15文字一致"
        return "Rank B", 0.70, match_detail

    # 類似度計算
    lev_score, jw_score = calc_similarity_scores(norm_t_a, norm_t_b)
    prefix_10 = (
        norm_t_a[:10] == norm_t_b[:10]
        if len(norm_t_a) >= 10 and len(norm_t_b) >= 10
        else False
    )

    # --- Rank C: レーベンシュタイン/Jaro-Winkler≧0.80 OR 先頭10文字一致 OR 1ブロック一致 ---
    if lev_score >= 0.80 or jw_score >= 0.80 or prefix_10 or block_matched_count >= 1:
        extra_info = f" [{block_detail}]" if block_matched_count == 1 else ""
        return (
            "Rank C",
            0.60,
            f"類似度(Lev:{lev_score:.2f}, JW:{jw_score:.2f}){extra_info}",
        )

    # --- Rank D: レーベンシュタイン≧0.70 OR 先頭7文字一致 ---
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

    # --- Rank E: 不一致 ---
    return "Rank E", 0.00, "不一致"


def process_matching(file_a_path, file_b_path):
    print("Excelファイルを読み込んでいます...")
    df_a = pd.read_excel(file_a_path)
    df_b = pd.read_excel(file_b_path)

    results = []

    print("照合処理を実行中（分割照合機能つき）...")
    for idx_a, row_a in df_a.iterrows():
        title_a = row_a.get("タイトル", "")
        author_a = row_a.get("著者", "")

        best_rank = "Rank E"
        best_score = 0.00
        best_match_row = None
        best_detail = ""

        rank_order = {
            "Rank SS": 6,
            "Rank S": 5,
            "Rank A": 4,
            "Rank B": 3,
            "Rank C": 2,
            "Rank D": 1,
            "Rank E": 0,
        }

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
                best_rank = rank
                best_score = score
                best_match_row = row_b
                best_detail = detail

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

    result_df = pd.DataFrame(results)
    output_filename = "match_result_advanced.xlsx"
    result_df.to_excel(output_filename, index=False)
    print(f"\n照合完了！結果を '{output_filename}' に保存しました。")
    return result_df


if __name__ == "__main__":
    FILE_A = "file_a.xlsx"
    FILE_B = "file_b.xlsx"

    if not os.path.exists(FILE_A) or not os.path.exists(FILE_B):
        print(
            f"エラー: {FILE_A} または {FILE_B} が見つかりません。ファイルを用意してください。"
        )
    else:
        result = process_matching(FILE_A, FILE_B)
        print("\n【照合結果プレビュー】")
        print(result[["ファイルA_タイトル", "判定ランク", "スコア", "マッチ詳細・条件"]])