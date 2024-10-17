from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.live import Live
from time import sleep
from tqdm import tqdm
import time

def splash_screen():
    console = Console()

    # ASCII Art for the Title (SMART PLUS) with Version below
    big_title = r"""
    ███████╗███╗   ███╗ █████╗ ██████╗ ████████╗               ██████╗ ██╗     ██╗   ██╗███████╗        
    ██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝               ██╔══██╗██║     ██║   ██║██╔════╝        
    ███████╗██╔████╔██║███████║██████╔╝   ██║                  ██████╔╝██║     ██║   ██║███████╗        
    ╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║                  ██╔═══╝ ██║     ██║   ██║╚════██║        
    ███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║                  ██║     ███████╗╚██████╔╝███████║        
    ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝                  ╚═╝     ╚══════╝ ╚═════╝ ╚══════╝        

                              Version: 1.0
    """

    # Subtitle (Bigger)
    subtitle = "[bold white]for E-Commerce Recommendations[/bold white]"

   

    # Big border/frame for the entire splash screen
    border_frame_top = "╔" + "═" * 88 + "╗"
    border_frame_bottom = "╚" + "═" * 88 + "╝"
    border_sides = "║"

    # Display the large ASCII art title with version
    console.print(f"[bold bright_cyan]{big_title}[/bold bright_cyan]\n", justify="center")
    console.print(subtitle, justify="center")  # Bigger subtitle

    # Display top border frame
    console.print(f"[bold white]{border_frame_top}[/bold white]", justify="center")
    
    # Display information in the middle with frame sides, larger font
    console.print(f"{border_sides}{' ' * 88}{border_sides}", justify="center")
    console.print(f"{border_sides}{'Developed by: Mohammed Abujayyab':^88}{border_sides}", style="bold white", justify="center")
    console.print(f"{border_sides}{'E-Mail: Moh.Abujayyab@gmail.com':^88}{border_sides}", style="bold white", justify="center")
    console.print(f"{border_sides}{'Berlin':^88}{border_sides}", style="bold white", justify="center")
    console.print(f"{border_sides}{'Copyright ©2024':^88}{border_sides}", style="bold white", justify="center")
    console.print(f"{border_sides}{' ' * 88}{border_sides}", justify="center")
    
    # Display bottom border frame
    console.print(f"[bold white]{border_frame_bottom}[/bold white]", justify="center")

    # Display Progress Bar (from tqdm)
    for i in tqdm(range(100), desc="Starting...", bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}"):
        time.sleep(0.02)  # Simulate loading with delay

    # Prompt user to press Enter to start
    console.print("\n[bold green]Press [white]Enter[green] to start the application...[/bold green]", justify="center")
    input()  # Wait for the user to press Enter
    console.print("Type 'python HybridRecSmartMLPlus' to use the application", justify="center")

if __name__ == "__main__":
    splash_screen()
