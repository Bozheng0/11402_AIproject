import sys
import os
import time

# --- 核心修正：將專案根目錄加入 Python 搜尋路徑 ---
# 這行能讓處於 tests/ 資料夾內的腳本順利找到 core/inference_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.inference_engine import FinalEvaluator

def run_test_suite():
    # 初始化模型
    print("正在從核心引擎載入模型，請稍候...")
    try:
        ev = FinalEvaluator()
    except Exception as e:
        print(f"❌ 引擎初始化失敗: {e}")
        return

    # 定義分類測試案例 (與你原本的內容一致)
    test_data = {
        "情感傳承類 (預期: 保留)": [
            "這是去世的爺爺親手削的鉛筆，我一直放在文具盒裡。",
            "這張照片雖然模糊，但紀錄了我們全家人第一次出國的樣子。",
            "初戀送我的乾花書籤，雖然已經碎了，但我不忍心扔掉。",
            "這件旗袍是媽媽年輕時參加婚禮穿的，很有紀念意義。"
        ],
        "身分成就類 (預期: 保留)": [
            "這是我大學畢業時拿到的校長獎獎牌。",
            "這是我為了創業跑的第一張訂單發票，對我來說很有意義。",
            "這套西裝是我第一次升職面試時穿的，它是我的幸運物。"
        ],
        "負面情緒與壓力類 (預期: 斷捨離)": [
            "這件衣服是前男友送的，每次看到它我就覺得很煩。",
            "這是我在上一間壓力很大的公司加班時穿的制服，充滿痛苦的回憶。",
            "這個杯子裂了一條縫，每次用它都提心吊膽。",
            "這是前任留下來的牙刷，看了就覺得礙眼。"
        ],
        "純粹囤積類 (預期: 斷捨離)": [
            "這個紙箱看起來很堅固，以後搬家說不定能用到。",
            "這堆舊數據線雖然暫時沒用，但萬一以後哪台設備要插呢？",
            "去年買的健身器材，雖然現在都拿來掛衣服，但我總覺得以後會練。"
        ],
        "雜物類 (預期: 斷捨離)": [
            "這是一個剛喝完的礦泉水瓶。",
            "去年的舊報紙，上面都是廣告。",
            "這是一支已經沒水的原子筆。"
        ]
    }

    results_summary = {"保留 (Keep)": 0, "建議斷捨離 (Discard)": 0}
    total_start_time = time.time()

    print("\n" + "="*60)
    print("🚀 開始執行斷捨離 AI 系統自動化測試 (v2.0 模組化版本)")
    print("="*60 + "\n")

    for category, sentences in test_data.items():
        print(f"【{category}】")
        for text in sentences:
            res = ev.evaluate(text)
            d = res['details']
            decision = res['decision']
            
            # 統計結果
            if "Keep" in decision or "保留" in decision:
                results_summary["保留 (Keep)"] += 1
                icon = "✅"
            else:
                results_summary["建議斷捨離 (Discard)"] += 1
                icon = "🗑️"

            print(f"{icon} 輸入: {text}")
            # 顯示情緒與信心度
            print(f"   -> 分數: {res['final_score']} | 情緒: {d['emotion']['label']} ({d['emotion']['conf']*100:.1f}%)")
            # 顯示馬斯洛與瑞斯動機
            print(f"   -> 心理: 需求={d['maslow']['label']} | 動機={d['reiss']['label']} | 強度={d['intensity']:.2f}")
            print(f"   -> 決策: {res['decision']}")
        print("-" * 40)

    total_duration = time.time() - total_start_time
    print("\n" + "="*60)
    print("📊 測試總結報告")
    print(f"總測試案例數: {sum(results_summary.values())}")
    print(f"保留個數: {results_summary['保留 (Keep)']}")
    print(f"斷捨離個數: {results_summary['建議斷捨離 (Discard)']}")
    print(f"總耗時: {total_duration:.2f} 秒")
    print("="*60)

if __name__ == "__main__":
    run_test_suite()