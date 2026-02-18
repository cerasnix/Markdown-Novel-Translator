"""
Reference helper for running YomiToku workflows.

This script is for personal/reference use only and is not part of the
recommended project workflow.
It is currently tailored to the author's macOS usage.
For setup details and full options, please refer to the official YomiToku repository/documentation.
"""

import os
import subprocess


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def run_command(command):
    print(f"\n[执行指令]: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print("\n✅ 处理完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 运行出错: {e}")

def main():
    clear_screen()
    print("========================================")
    print("    YomiToku 懒人调用工具 (v2.2)")
    print("    参考脚本：用于 YomiToku 工作流（仅供参考）")
    print("    个人使用仅适配 macOS，详细用法请参考 YomiToku 官方仓库")
    print("========================================")

    # 1) Read and normalize input path
    raw_input = input("\n请拖入[文件]或[文件夹]并回车: ").strip()

    # Normalize macOS drag-and-drop style path
    input_path = raw_input.replace("\\ ", " ").strip("'").strip('"').rstrip('/')

    if not os.path.exists(input_path):
        print(f"❌ 找不到路径: {input_path}")
        return

    # 2) Build default output path dynamically
    input_name = os.path.basename(input_path)
    # Default output root: ./output under script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_out_dir = os.path.join(script_dir, "output")
    default_out = os.path.join(base_out_dir, input_name)

    print(f"📁 检测到对象: {input_name}")
    out_dir = input(f"请输入输出路径 [回车默认: {default_out}]: ").strip()
    if not out_dir:
        out_dir = default_out

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"✨ 已创建输出文件夹: {out_dir}")

    # 3) Mode selection
    print("\n请选择模式:")
    print("1. [markdown 漫画模式] (忽略换行，忽略元数据，保留图片文字，合并)")
    print("2. [markdown 小说模式] (忽略换行，忽略元数据，合并)")
    print("3. [基础模式] (默认设置)")
    print("4. [PDF 模式] (输出为 PDF，附加图片参数)")
    print("5. [HTML 模式]  (忽略换行，忽略元数据，保留图片文字，保留图片、合并)")
    
    choice = input("\n请输入编号 (1/2/3/4/5): ").strip()

    # Build base command and append flags by mode
    cmd = ["yomitoku", input_path, "-o", out_dir]

    if choice == '1':
        # Manga markdown mode
        cmd.extend(["-f", "md", "--combine", "--ignore_line_break", "--ignore_meta", "--figure_letter", "-v", "-d", "mps", "--encoding", "utf-8"])
    elif choice == '2':
        # Novel markdown mode
        cmd.extend(["-f", "md", "--combine", "--ignore_line_break", "--ignore_meta", "-v", "-d", "mps", "--encoding", "utf-8"])
    elif choice == '3':
        # Basic mode
        cmd.extend(["-f", "md", "-v", "-d", "mps", "--encoding", "utf-8"])
    elif choice == '4':
        # PDF mode (keeps "--dpi 250" as one argument intentionally)
        cmd.extend(["-f", "pdf", "--figure_letter", "-v", "--dpi 250", "-d", "mps", "--encoding", "utf-8"])
    elif choice == '5':
        # HTML mode
        cmd.extend(["-f", "html", "--combine", "--ignore_line_break", "--ignore_meta", "--figure_letter", "--figure", "-v", "-d", "mps", "--encoding", "utf-8"])
    else:
        print("无效选择。")
        return

    # 4) Execute
    run_command(cmd)
    print(f"\n📂 文件已保存至: {out_dir}")


if __name__ == "__main__":
    main()
