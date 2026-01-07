import os
import yaml
import shutil

from pathlib import Path


def initial_config(config_path):
    # 读取 YAML 文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config["t_max"] = 7_000_000

    # 写回 YAML 文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True)

def main():
    current_file_path = Path(__file__).parent

    # 复制初始参数文件
    shutil.copy(
        current_file_path / "src" / "config" / "algs" / "ant_metaq_v5_bak.yaml",
        current_file_path / "src" / "config" / "algs" / "ant_metaq_v5.yaml",
    )
    initial_config(
        current_file_path / "src" / "config" / "algs" / "ant_metaq_v5.yaml"
    )
    os.system(f"python src/main.py --config=ant_metaq_v5 --env-config=sc2 with env_args.map_name=25m")

if __name__ == "__main__":
    main()

