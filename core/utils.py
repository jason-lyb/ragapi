from jamo import h2j, j2hcj

def split_hangul(text: str) -> str:
    """'베드로' => 'ㅂㅔㄷㅡㄹㅗ'"""
    return ''.join(j2hcj(h2j(text)))

