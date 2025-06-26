import os
import subprocess
from pathlib import Path

# 配置部分
PROJECT_ROOT = Path(__file__).parent  # 默认是脚本所在目录
IGNORE_DIRS = {'.git', '__pycache__', 'venv', '.venv', 'env', '.idea', 'node_modules'}
IGNORE_FILES = {'.pyc', '.pyo', '.DS_Store'}

def generate_tree(start_path):
    """生成树状结构字符串"""
    tree_str = ''

    for root, dirs, files in os.walk(start_path):
        # 过滤掉忽略的目录
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        level = root.replace(str(start_path), '').count(os.sep)
        indent = '│   ' * (level)
        relative_root = os.path.relpath(root, start_path)

        if level == 0:
            tree_str += f'{os.path.basename(root)}/\n'
        else:
            tree_str += f'{indent}├── {os.path.basename(root)}/\n'

        sub_indent = '│   ' * (level + 1)
        for file in sorted(files):
            if any(file.endswith(ext) for ext in IGNORE_FILES):
                continue
            tree_str += f'{sub_indent}├── {file}\n'

    return tree_str.rstrip('\n')

def write_requirements_file(filename='requirements.txt'):
    """获取当前环境依赖并写入 requirements.txt"""
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE, text=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(result.stdout)
    print(f"✅ 已生成 {filename}")

def write_structure_file(tree_str, filename='PROJECT_STRUCTURE.md'):
    """写入结构到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('```\n')
        f.write(tree_str)
        f.write('\n```\n')
    print(f"✅ 已生成 {filename}")

if __name__ == '__main__':
    structure = generate_tree(PROJECT_ROOT)
    print("📋 项目结构如下：")
    print(structure)

    # 写入结构文件
    write_structure_file(structure)

    # 生成 requirements.txt
    write_requirements_file()