class FilterCodeGenerator:
    def __init__(self):
        self._code_rows = [
            "import cv2",
            "",
            "# ファイル読み込み",
            "img = cv2.imread(ファイルパス)"
        ]

    def add(self, code: str):
        self._code_rows.extend(code.split("\n") + [""])

    def generate(self):
        return "\n".join(self._code_rows)

    def arranged_row_count(self):
        return len(self._code_rows)


class FilterChainCodeGenerator:
    def __init__(self):
        self._code_rows = [
            "import cv2",
            "",
            "# ファイル読み込み",
            "img = cv2.imread(ファイルパス)"
        ]

    def add(self, code: str):
        self._code_rows.extend(code.split("\n") + [""])

    def generate(self):
        return "\n".join(self._code_rows)

    def row_count(self):
        return len(self._code_rows)
