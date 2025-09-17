"""Lightweight language helpers shared by agents."""


def detect_lang(s: str) -> str:
    """Crude detection: if any CJK character exists → 'zh', else 'en'.
    (Exact logic extracted from ScribeAgent._detect_lang)
    """
    try:
        for ch in s:
            if '\u4e00' <= ch <= '\u9fff':
                return 'zh'
    except Exception:
        pass
    return 'en'

