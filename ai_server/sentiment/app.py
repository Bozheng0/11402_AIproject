from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from sentiment.core.inference_engine import FinalEvaluator

console = Console()

def display_report(res):
    color = "green" if res['final_score'] >= 60 else "red"
    
    # 建立精美的結果表格
    table = Table(title="🧠 AI 心理多維度分析", show_header=True, header_style="bold cyan")
    table.add_column("分析維度", style="dim")
    table.add_column("識別結果")
    table.add_column("信心度 / 強度", justify="right")

    # 呈現各維度的信心分數
    d = res['details']
    table.add_row("當下情緒 (GoEmo)", d['emotion']['label'], f"{d['emotion']['conf']*100:.1f}%")
    table.add_row("深層需求 (Maslow)", d['maslow']['label'], f"{d['maslow']['conf']*100:.1f}%")
    table.add_row("情感強度 (SemEval)", "Intensity", f"{d['intensity']:.2f}")

    # 主面板
    console.print(Panel(
        f"[bold]綜合決策分數:[/] [bold cyan]{res['final_score']}[/]/100\n"
        f"[bold]最終建議:[/] [bold {color}]{res['decision']}[/]",
        title="📊 評估報告", 
        border_style=color
    ))
    console.print(table)

def main():
    console.print(Panel.fit("🏠 [bold blue]AI 斷捨離心理決策系統[/] [dim]v2.0[/]\n[italic]請輸入物品描述，輸入 'exit' 退出[/]", border_style="blue"))
    
    # 載入模型 (顯示進度條比較帥)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="正在初始化三位一體 AI 模型...", total=None)
        ev = FinalEvaluator()
    
    while True:
        user_input = console.input("[bold yellow]請描述你的物品: [/]")
        
        if user_input.lower() == 'exit':
            console.print("[bold red]系統關閉。祝你斷捨離順利！[/]")
            break
            
        if not user_input.strip():
            continue
            
        with console.status("[bold green]AI 正在分析心理動機..."):
            try:
                results = ev.evaluate(user_input)
                display_report(results)
            except Exception as e:
                console.print(f"[bold red]錯誤:[/] {e}")

if __name__ == "__main__":
    main()