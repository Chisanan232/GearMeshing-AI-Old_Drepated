from __future__ import annotations  # PEP 563/649：推遲型別解析

import os
import pathlib
import time
from typing import Final  # PEP 591: final names

from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

# ------- 常量設定 -------
ROOT_DIR: Final[pathlib.Path] = pathlib.Path("./")
print(f"[DEBUG] ROOT_DIR: {ROOT_DIR}")
WORKSPACE: Final[pathlib.Path] = ROOT_DIR / "test" / "workspace"
print(f"[DEBUG] WORKSPACE: {WORKSPACE}")
WORKSPACE.mkdir(exist_ok=True)

# 載入環境變數從 .env 檔案
load_dotenv(ROOT_DIR / ".env")

# 設置超時時間（秒）
TIMEOUT = 60
# Set the rounds of having conversation.
MAX_TURNS = 5

# ------- LLM 配置 -------
LLM_CONFIG = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.environ["OPENAI_API_KEY"],
        }
    ],
    "timeout": TIMEOUT,
}

# ------- 建立代理 -------
# 創建助手代理 - 專注於撰寫高品質代碼的 Python 專家
assistant = AssistantAgent(
    name="assistant",
    llm_config=LLM_CONFIG,
    system_message="""你是一名 Python 專家，專精於撰寫高品質的 Python 程式碼。
    你的目標是幫助用戶解決程式設計問題，編寫高效能、易於理解且符合最佳實踐的代碼。

    請遵循 PEP8 規範，使用型別註解，並優先使用內建的泛型（如 list[int] 而非 typing.List[int]）。

    當用戶要求你創建檔案時，請使用實際的 Python 文件操作程式碼來建立檔案，而不是只返回代碼片段。
    始終確保功能完整、測試周全，並優先考慮程式碼的安全性和可維護性。
    """,
)

# 創建使用者代理，負責執行程式碼
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # 全自動運行
    code_execution_config={
        "work_dir": str(WORKSPACE.absolute()),
        "use_docker": False,  # 直接在本地環境執行
        "timeout": TIMEOUT,  # 設置程式碼執行超時
    },
    llm_config=LLM_CONFIG,  # 讓 user_proxy 也能生成回應
    system_message="""你的任務是執行 Python 程式碼並報告結果。
    當需要建立檔案時，請確保使用 Python 的檔案操作函數（如 open(), write() 等）來實際建立和寫入檔案，
    而不是只展示代碼。

    請保持回應簡短、具體，集中在成功或失敗的結果。如果執行失敗，分析問題並嘗試修復。
    """,
)


def main() -> None:
    print("[INFO] 啟動 AI 代理來解決程式設計任務...")

    start_time = time.time()

    try:
        # 開始對話，讓 AI 自行解決問題
        user_proxy.initiate_chat(
            assistant,
            message="""我需要一個簡單的階乘函數和它的單元測試。請幫我：

    1. 建立名為 math_utils.py 的檔案，實作一個 factorial 函數，可計算非負整數的階乘
       - 請包含適當的型別註解和文件字串
       - 記得處理邊界情況和錯誤輸入

    2. 建立名為 test_math_utils.py 的檔案，使用 unittest 模組編寫完整的單元測試
       - 請測試各種案例，包括邊界情況

    3. 執行單元測試，確保所有測試都通過

    請直接開始解決這個任務，不需要詳細解釋你要如何處理。重點是用程式碼解決問題，並確保檔案被實際建立和儲存。""",
            max_turns=MAX_TURNS,  # 限制對話輪數，避免無限循環
        )

    except Exception as e:
        print(f"[ERROR] 執行過程出錯: {e}")
    finally:
        # 顯示總執行時間
        elapsed_time = time.time() - start_time
        print(f"[INFO] 總執行時間: {elapsed_time:.2f} 秒")


# ------- 啟動對話 -------
if __name__ == "__main__":
    main()
