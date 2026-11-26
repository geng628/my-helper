"""中英双语文本分词与轻量词干归一化底座（零内部依赖）。"""
from __future__ import annotations

import re

_ASCII_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

# 高频英文功能词/疑问词停用表，避免长对话检索被 what/did/the 等噪声淹没
EN_STOPWORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "all", "also", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
        "done", "down", "during", "each", "else", "few", "for", "from", "further",
        "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
        "just", "may", "me", "might", "more", "most", "must", "my", "myself", "no",
        "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
        "ours", "out", "over", "own", "s", "same", "shall", "she", "should", "so",
        "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "then",
        "there", "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
        "while", "who", "whom", "whose", "why", "will", "with", "would", "you", "your",
        "yours", "tell", "like", "know", "think", "really", "actually", "still",
    }
)

# 常见英文不规则动词词形还原映射，提升英文多会话问答召回
EN_IRREGULAR_LEMMAS: dict[str, str] = {
    "went": "go",
    "gone": "go",
    "bought": "buy",
    "ran": "run",
    "made": "make",
    "gave": "give",
    "given": "give",
    "took": "take",
    "taken": "take",
    "saw": "see",
    "seen": "see",
    "told": "tell",
    "found": "find",
    "won": "win",
    "met": "meet",
    "wrote": "write",
    "written": "write",
    "ate": "eat",
    "eaten": "eat",
    "drove": "drive",
    "driven": "drive",
    "spoke": "speak",
    "spoken": "speak",
    "chose": "choose",
    "chosen": "choose",
    "built": "build",
    "sent": "send",
    "spent": "spend",
    "left": "leave",
    "kept": "keep",
    "felt": "feel",
    "heard": "hear",
    "held": "hold",
    "read": "read",
    "children": "child",
    "people": "person",
    "women": "woman",
    "men": "man",
}

# 中文疑问词/连接虚词切分正则（用于查询侧去噪与双侧轻度虚词切分）
_CJK_QUERY_NOISE_RE = re.compile(
    r"(?:有什么|是什么|是多少|为什么|怎么样|是怎么|怎么|怎样|如何|哪些|哪里|哪个|哪天|哪一|"
    r"什么|是否|能否|可以|请问|一下|以及|或者|还是|但是|如果|的话|之后|之前|里面|有关|关于|"
    r"的|了|吗|呢|啊|吧|和|与|及)"
)
_CJK_DOC_PARTICLE_RE = re.compile(r"[的了吗呢啊吧]")


# 对单个英文词执行轻量词干与不规则动词归一化（零依赖，保护 macos 等非复数词尾）
def _normalize_en_word(word: str) -> str:
    if word in EN_IRREGULAR_LEMMAS:
        return EN_IRREGULAR_LEMMAS[word]
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) >= 5:
        return word[:-3] + "y"
    if word.endswith("ied") and len(word) >= 5:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) >= 6:
        stem = word[:-3]
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in {"s", "l", "z"}:
            stem = stem[:-1]
        return stem
    if word.endswith("ed") and len(word) >= 5:
        stem = word[:-2]
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in {"s", "l", "z"}:
            stem = stem[:-1]
        return stem
    if word.endswith(("ches", "shes", "xes", "zes", "sses")) and len(word) >= 6:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is", "os", "as", "js", "ts", "rs")):
        return word[:-1]
    return word


# 对中英文混合文本做统一分词：英文去停用词+词干归一化，中文按虚词切段并提取双字 bigram
def tokenize_text(text: str, *, for_query: bool = False) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []

    for word in _ASCII_WORD_RE.findall(lowered):
        if len(word) < 2 and not word.isdigit():
            continue
        if word in EN_STOPWORDS:
            continue
        norm = _normalize_en_word(word)
        if norm in EN_STOPWORDS:
            continue
        tokens.append(norm)

    splitter = _CJK_QUERY_NOISE_RE if for_query else _CJK_DOC_PARTICLE_RE
    for run in _CJK_RUN_RE.findall(lowered):
        for seg in splitter.split(run):
            if len(seg) >= 2:
                for idx in range(len(seg) - 1):
                    tokens.append(seg[idx : idx + 2])
            elif len(seg) == 1 and not for_query:
                tokens.append(seg)

    # 兜底：若查询全部由停用词或单字构成，保留原始有效字符防止空查询报错
    if for_query and not tokens:
        for word in _ASCII_WORD_RE.findall(lowered):
            if word:
                tokens.append(word)
        for run in _CJK_RUN_RE.findall(lowered):
            if len(run) >= 2:
                for idx in range(len(run) - 1):
                    tokens.append(run[idx : idx + 2])
            elif run:
                tokens.append(run)
    return tokens
