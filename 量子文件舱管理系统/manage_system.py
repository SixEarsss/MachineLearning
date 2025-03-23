import os # 导入os模块，用于与操作系统交互，如文件系统操作
import shutil # 导入shutil模块，提供文件和目录的高级操作
from datetime import datetime # 导入datetime模块，用于处理时间和日期
import json # 导入json模块，用于处理json格式的数据


# 定义文件拓展名与目标目录的映射关系
structure = {
    '.quantum': ('quantum_core', 'SECTOR-7G'),
    '.holo': ('hologram_vault', 'CHAMBER-12F'),
    '.exo': ('exobiology_lab', 'POD-09X'),
    '.chrono': ('temporal_archive', 'VAULT-00T')
}

# 定义文件分类的函数
def classify_files(directory):
    # 遍历目录中的文件
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file) # 获取文件的完整路径

        if not os.path.isfile(file_path): # 如果路径不是文件，跳过
            continue

        ext = os.path.splitext(file)[1] # 将文件名拆分为文件名和拓展名两部分，获取文件拓展名
        dest_info = structure.get(ext)
        if dest_info:
            dest_dir = os.path.join(*dest_info)
        else:
            dest_dir = "quantum_quarantine"


        if not dest_info: # 如果是未知类型的文件
            new_file_name = f"ENCRYPTED_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file}" # 进行重命名（前缀加 ENCRYPTED_）并加上时间戳
            dest_path = os.path.join(directory, dest_dir, new_file_name) # 生成目标路径
        else: # 如果是已知类型的文件
            new_file_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file}" # 同样为文件命名，添加时间戳
            dest_path = os.path.join(directory,dest_dir,new_file_name) # 生成目标路径

        os.makedirs(os.path.dirname(dest_path), exist_ok=True) # 创建目标文件夹，若已存在则不做任何操作

        shutil.move(file_path, dest_path) # 将文件移动到目标目录
        print(f"移动 {file} 至 {dest_path}") # 打印文件移动的信息

# 定义生成日志文件的函数
def generate_log(directory):
    # 日志头部
    header = [
        "┌──────────────────────────────┐",
        "│ 🛸 Xia-III 空间站数据分布全息图 │",
        "└──────────────────────────────┘",
        ""
    ]
    # 日志尾部
    footer = [
        "",
        f"🤖 SuperNova · 地球标准时  {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        "⚠️ 警告：请勿直视量子文件核心"
        
    ]

    # 日志文件的路径
    log_file = os.path.join(directory,"log.txt")
    
    # 打开日志文件进行写入操作
    with open(log_file, "w",encoding="utf-8") as log:
        log.write("\n".join(header) + "\n")  # 写入头部
        
        # 写入日志中间部分信息
        # 遍历所有文件
        for root,_,files in os.walk(directory):
            level = root.replace(directory, "").count(os.sep) # 计算当前目录的层级
            indent = "│   " * level + "├─ " # 计算当前层级所对应的缩进
            log.write(f"{indent}🚀 {os.path.basename(root)}\n") # 写入当前目录的名称
            for file in files: # 遍历当前目录下的文件
                file_indent = "│   " * (level + 1) + "├─ " # 计算当前文件所对应的缩进 
                symbol = "🔮" if "ENCRYPTED" not in file else "⚠️" # 如果文件没有“ENCRYPTED”则用🔮标记，否则用⚠️
                log.write(f"{file_indent}{symbol} {file}\n") # 写入文件信息
        log.write("\n".join(footer) + "\n")  # 写入尾部

    return log_file # 返回生成的日志文件的路径

def generate_json_log(directory, log_file):
    # 读取日志文件内容
    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.readlines()  # 读取所有行

    # 生成 JSON 格式的日志
    json_log = json.dumps({"log_content": log_content}, indent=4, ensure_ascii=False)

    json_log_file = os.path.join(directory, "log.json")  # JSON 文件路径
    with open(json_log_file, "w", encoding="utf-8") as log:
        log.write(json_log)

    print(f"生成JSON日志的路径为：{json_log_file}")  # 打印 JSON 日志的路径



def generate_test_input(directory):
    # 使用题干中给的示例文件名
    files = [
        "alien_research.quantum",
        "unknown_species.exo",
        "mystery_signal.chrono",
        "imsb.xyz"
    ]
    
    # 创建目录（如果不存在）
    os.makedirs(directory, exist_ok=True)
    
    # 生成文件
    for file in files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'w') as f:
            f.write(f"Sample content for {file}\n")
    
    print(f"已在 '{directory}' 中生成以下文件：")
    for file in files:
        print(f" - {file}")

# 示例用法

# 定义主函数
def main():
    incoming_dir = "incoming_data" # 设置输入数据的目录
    generate_test_input(incoming_dir)
    classify_files(incoming_dir) # 使用分类文件函数进行文件分类并移动到目标目录
    log_file = generate_log(incoming_dir) # 生成日志文件
    generate_json_log(incoming_dir,log_file) # 生成JSON日志文件

# 执行主函数
if __name__ == "__main__":
    main()
    