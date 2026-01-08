import os
import yaml
import shutil

from pathlib import Path


def get_latest_checkpoint(parent_dir):
    folders = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)
               if os.path.isdir(os.path.join(parent_dir, f))]
    
    if not folders:
        return None  # 没有子文件夹

    # 获取每个文件夹的创建时间
    folders_with_ctime = [(folder, os.path.getctime(folder)) for folder in folders]
    
    # 按创建时间排序，取最新的
    latest_folder, _ = max(folders_with_ctime, key=lambda x: x[1])
    return latest_folder


def change_single_thread(config_path):
    # 读取 YAML 文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config["runner"] = "episode"
    config["batch_size_run"] = 1

    # 写回 YAML 文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True)

def change_default_yaml(config_path):
    # 读取 YAML 文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    current_file_path = Path(__file__).parent
    model_path = (current_file_path / "results" / "models").resolve()
    config["checkpoint_path"] = get_latest_checkpoint(model_path)
    config["evaluate"] = True
    config["save_replay"] = True

    # 写回 YAML 文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True)


def main():
    current_file_path = Path(__file__).parent

    env_lists = [
        "21m", "22m", "23m", "24m", "25m",
        "26m", "27m", "28m", "29m", "30m"
    ]
    env_index = -1

    # 复制初始参数文件
    shutil.copy(
        current_file_path / "src" / "config" / "algs" / "ant_metaq_v5_bak.yaml",
        current_file_path / "src" / "config" / "algs" / "ant_metaq_v5.yaml",
    )
    shutil.copy(
        current_file_path / "src" / "config" / "default_backup.yaml",
        current_file_path / "src" / "config" / "default.yaml",
    )
    change_single_thread(
        current_file_path / "src" / "config" / "algs" / "ant_metaq_v5.yaml"
    )
    change_default_yaml(
        current_file_path / "src" / "config" / "default.yaml",
    )

    while True:
        env_index += 1
        env_index %= len(env_lists)

        os.system(f"python src/main.py --config=ant_metaq_v5 --env-config=sc2 with env_args.map_name={env_lists[env_index]}")
        
        if env_index >= 9:
            break


if __name__ == "__main__":
    main()