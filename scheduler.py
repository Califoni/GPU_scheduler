#!/usr/bin/env python3
"""
GPU Training Task Scheduler
自动检测 GPU 占用情况并调度训练任务
"""

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml

# 配置路径
SCHEDULER_DIR = Path(__file__).parent
CONFIG_FILE = SCHEDULER_DIR / "config.yaml"
POOLS_DIR = SCHEDULER_DIR / "pools"
LOGS_DIR = SCHEDULER_DIR / "logs"

# 任务池文件
PENDING_FILE = POOLS_DIR / "pending.json"
RUNNING_FILE = POOLS_DIR / "running.json"
COMPLETED_FILE = POOLS_DIR / "completed.json"
FAILED_FILE = POOLS_DIR / "failed.json"

# 调度间隔（秒）
SCHEDULE_INTERVAL = 300  # 5 分钟

# GPU 空闲判断阈值（MB）
GPU_FREE_THRESHOLD_MB = 100


@dataclass
class Task:
    """任务数据类"""
    name: str
    work_dir: str
    docker_script: str
    container_work_dir: str
    train_command: str
    gpu_count: int
    completion_file: str
    # 运行时信息
    assigned_gpus: Optional[List[int]] = None
    pid: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    # 配置追踪
    config_hash: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        # 处理旧数据兼容性
        data.setdefault('config_hash', None)
        data.setdefault('retry_count', 0)
        return cls(**data)


def setup_logging() -> logging.Logger:
    """设置日志"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("gpu_scheduler")
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(
        LOGS_DIR / "scheduler.log",
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# ============== GPU 检测模块 ==============

def get_gpu_memory_usage() -> Dict[int, float]:
    """
    获取每张 GPU 的显存占用（MB）
    返回: {gpu_id: memory_used_mb}
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )

        gpu_memory = {}
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(",")
                gpu_id = int(parts[0].strip())
                memory_used = float(parts[1].strip())
                gpu_memory[gpu_id] = memory_used

        return gpu_memory
    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi 执行失败: {e}")
        return {}
    except Exception as e:
        logger.error(f"获取 GPU 信息失败: {e}")
        return {}


def get_free_gpus() -> List[int]:
    """
    获取空闲 GPU 列表
    空闲标准：显存占用 < GPU_FREE_THRESHOLD_MB
    """
    gpu_memory = get_gpu_memory_usage()
    free_gpus = [
        gpu_id for gpu_id, memory in gpu_memory.items()
        if memory < GPU_FREE_THRESHOLD_MB
    ]
    return sorted(free_gpus)


def allocate_gpus(free_gpus: List[int], count: int) -> Optional[List[int]]:
    """
    分配 GPU，优先分配连续的 GPU
    返回分配的 GPU ID 列表，如果无法分配则返回 None
    """
    if len(free_gpus) < count:
        return None

    # 尝试找连续的 GPU
    for i in range(len(free_gpus) - count + 1):
        candidate = free_gpus[i:i + count]
        # 检查是否连续
        if candidate[-1] - candidate[0] == count - 1:
            return candidate

    # 如果没有连续的，返回前 count 个
    return free_gpus[:count]


# ============== 任务池管理模块 ==============

def load_pool(pool_file: Path) -> List[Task]:
    """从 JSON 文件加载任务池"""
    if not pool_file.exists():
        return []

    try:
        with open(pool_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [Task.from_dict(task) for task in data]
    except Exception as e:
        logger.error(f"加载任务池 {pool_file} 失败: {e}")
        return []


def save_pool(pool_file: Path, tasks: List[Task]) -> None:
    """保存任务池到 JSON 文件"""
    pool_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(pool_file, "w", encoding="utf-8") as f:
            json.dump([task.to_dict() for task in tasks], f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"保存任务池 {pool_file} 失败: {e}")



def compute_config_hash(task_config: Dict[str, Any]) -> str:
    """计算任务配置的哈希值，用于检测配置变更"""
    core_fields = ['work_dir', 'docker_script', 'container_work_dir',
                   'train_command', 'gpu_count', 'completion_file']
    hash_content = {k: task_config.get(k) for k in core_fields}
    return hashlib.md5(json.dumps(hash_content, sort_keys=True).encode()).hexdigest()[:8]


def load_config_tasks_raw() -> List[Dict[str, Any]]:
    """从配置文件加载任务（保留原始字典格式）"""
    if not CONFIG_FILE.exists():
        logger.warning(f"配置文件 {CONFIG_FILE} 不存在")
        return []

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config.get("tasks", [])
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return []


def create_task_from_config(task_config: Dict[str, Any], config_hash: str) -> Task:
    """从配置字典创建 Task 对象"""
    return Task(
        name=task_config["name"],
        work_dir=task_config["work_dir"],
        docker_script=task_config["docker_script"],
        container_work_dir=task_config["container_work_dir"],
        train_command=task_config["train_command"],
        gpu_count=task_config["gpu_count"],
        completion_file=task_config["completion_file"],
        config_hash=config_hash
    )


def load_config_tasks() -> List[Task]:
    """从配置文件加载任务"""
    if not CONFIG_FILE.exists():
        logger.warning(f"配置文件 {CONFIG_FILE} 不存在")
        return []

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        tasks = []
        for task_config in config.get("tasks", []):
            task = Task(
                name=task_config["name"],
                work_dir=task_config["work_dir"],
                docker_script=task_config["docker_script"],
                container_work_dir=task_config["container_work_dir"],
                train_command=task_config["train_command"],
                gpu_count=task_config["gpu_count"],
                completion_file=task_config["completion_file"]
            )
            tasks.append(task)

        return tasks
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return []


def sync_pending_pool() -> None:
    """
    同步 pending 池与配置文件
    - 更新pending池中配置已变更的任务
    - 处理retry标志：将failed任务移回pending
    - 处理force_rerun标志：将completed任务移回pending
    - 处理enabled标志：禁用任务从pending移除
    - 移除配置文件中已删除的任务
    - 添加配置中新增的任务
    """
    config_tasks_raw = load_config_tasks_raw()
    pending_tasks = load_pool(PENDING_FILE)
    running_tasks = load_pool(RUNNING_FILE)
    completed_tasks = load_pool(COMPLETED_FILE)
    failed_tasks = load_pool(FAILED_FILE)

    # 构建配置任务的映射 {name: (config_dict, hash)}
    config_map: Dict[str, tuple] = {}
    for task_config in config_tasks_raw:
        name = task_config.get("name")
        if name:
            config_hash = compute_config_hash(task_config)
            config_map[name] = (task_config, config_hash)

    # 构建各池的任务名集合
    pending_names = {t.name for t in pending_tasks}
    running_names = {t.name for t in running_tasks}
    completed_names = {t.name for t in completed_tasks}
    failed_names = {t.name for t in failed_tasks}

    pending_modified = False
    completed_modified = False
    failed_modified = False

    # 1. 更新pending池中配置已变更的任务
    new_pending_tasks = []
    for task in pending_tasks:
        if task.name not in config_map:
            # 配置中已删除的任务，从pending移除
            logger.info(f"任务 {task.name} 已从配置中删除，从 pending 池移除")
            pending_modified = True
            continue

        task_config, config_hash = config_map[task.name]
        enabled = task_config.get("enabled", True)

        if not enabled:
            # 禁用的任务从pending移除
            logger.info(f"任务 {task.name} 已禁用，从 pending 池移除")
            pending_modified = True
            continue

        # 检查配置是否变更
        if task.config_hash != config_hash:
            # 配置已变更，更新任务
            updated_task = create_task_from_config(task_config, config_hash)
            updated_task.retry_count = task.retry_count
            new_pending_tasks.append(updated_task)
            logger.info(f"任务 {task.name} 配置已更新 (hash: {task.config_hash} -> {config_hash})")
            pending_modified = True
        else:
            new_pending_tasks.append(task)

    pending_tasks = new_pending_tasks

    # 2. 处理retry标志：将failed任务移回pending
    new_failed_tasks = []
    for task in failed_tasks:
        if task.name not in config_map:
            new_failed_tasks.append(task)
            continue

        task_config, config_hash = config_map[task.name]
        retry = task_config.get("retry", False)

        if retry:
            # 重置任务状态并移回pending
            retried_task = create_task_from_config(task_config, config_hash)
            retried_task.retry_count = task.retry_count + 1
            pending_tasks.append(retried_task)
            logger.info(f"任务 {task.name} 设置了 retry=true，从 failed 池移回 pending 池 (重试次数: {retried_task.retry_count})")
            pending_modified = True
            failed_modified = True
        else:
            new_failed_tasks.append(task)

    failed_tasks = new_failed_tasks

    # 3. 处理force_rerun标志：将completed任务移回pending
    new_completed_tasks = []
    for task in completed_tasks:
        if task.name not in config_map:
            new_completed_tasks.append(task)
            continue

        task_config, config_hash = config_map[task.name]
        force_rerun = task_config.get("force_rerun", False)

        if force_rerun:
            # 重置任务状态并移回pending
            rerun_task = create_task_from_config(task_config, config_hash)
            rerun_task.retry_count = task.retry_count
            pending_tasks.append(rerun_task)
            logger.info(f"任务 {task.name} 设置了 force_rerun=true，从 completed 池移回 pending 池")
            pending_modified = True
            completed_modified = True
        else:
            new_completed_tasks.append(task)

    completed_tasks = new_completed_tasks

    # 4. 添加新任务到pending池
    existing_names = pending_names | running_names | completed_names | failed_names
    for name, (task_config, config_hash) in config_map.items():
        if name not in existing_names:
            enabled = task_config.get("enabled", True)
            if enabled:
                new_task = create_task_from_config(task_config, config_hash)
                pending_tasks.append(new_task)
                logger.info(f"新任务添加到 pending 池: {name}")
                pending_modified = True

    # 保存修改后的池
    if pending_modified:
        save_pool(PENDING_FILE, pending_tasks)
    if failed_modified:
        save_pool(FAILED_FILE, failed_tasks)
    if completed_modified:
        save_pool(COMPLETED_FILE, completed_tasks)


# ============== 任务执行模块 ==============

def start_task(task: Task, gpus: List[int]) -> Optional[int]:
    """
    启动任务
    返回进程 PID，失败返回 None
    """
    gpu_str = ",".join(map(str, gpus))

    # 构建完整的 Docker 脚本路径
    docker_script_path = os.path.join(task.work_dir, task.docker_script)

    # 构建容器内执行的命令
    container_command = f"cd {task.container_work_dir} && {task.train_command}"

    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str

    try:
        # 启动 Docker 并执行训练命令
        # 假设 docker_run.sh 接受命令作为参数
        process = subprocess.Popen(
            ["bash", docker_script_path, container_command],
            env=env,
            cwd=task.work_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # 使进程在后台独立运行
        )

        logger.info(f"任务 {task.name} 已启动, PID: {process.pid}, GPUs: {gpu_str}")
        return process.pid

    except Exception as e:
        logger.error(f"启动任务 {task.name} 失败: {e}")
        return None


def is_process_running(pid: int) -> bool:
    """检查进程是否仍在运行"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def check_completion_file(filepath: str) -> bool:
    """检查完成标志文件是否存在"""
    return os.path.exists(filepath)


# ============== 任务监控模块 ==============

def monitor_running_tasks() -> None:
    """
    监控运行中的任务
    - 完成文件存在 → 移到 completed 池
    - 进程退出且无完成文件 → 移到 failed 池
    """
    running_tasks = load_pool(RUNNING_FILE)
    completed_tasks = load_pool(COMPLETED_FILE)
    failed_tasks = load_pool(FAILED_FILE)

    still_running = []

    for task in running_tasks:
        completion_exists = check_completion_file(task.completion_file)
        process_running = is_process_running(task.pid) if task.pid else False

        if completion_exists:
            # 任务完成
            task.end_time = datetime.now().isoformat()
            completed_tasks.append(task)
            logger.info(f"任务 {task.name} 已完成")
        elif not process_running:
            # 进程退出但没有完成文件，视为失败
            task.end_time = datetime.now().isoformat()
            failed_tasks.append(task)
            logger.warning(f"任务 {task.name} 失败（进程退出但无完成文件）")
        else:
            # 仍在运行
            still_running.append(task)

    # 保存更新后的任务池
    save_pool(RUNNING_FILE, still_running)
    save_pool(COMPLETED_FILE, completed_tasks)
    save_pool(FAILED_FILE, failed_tasks)


# ============== 任务调度模块 ==============

def schedule_tasks() -> None:
    """
    调度任务
    - 获取空闲 GPU
    - 从 pending 池选择可运行的任务
    - 分配 GPU 并启动任务
    """
    # 获取当前运行中任务占用的 GPU
    running_tasks = load_pool(RUNNING_FILE)
    occupied_gpus = set()
    for task in running_tasks:
        if task.assigned_gpus:
            occupied_gpus.update(task.assigned_gpus)

    # 获取空闲 GPU（排除已分配的）
    free_gpus = [g for g in get_free_gpus() if g not in occupied_gpus]

    if not free_gpus:
        logger.info("没有空闲 GPU")
        return

    logger.info(f"空闲 GPU: {free_gpus}")

    # 加载 pending 任务
    pending_tasks = load_pool(PENDING_FILE)

    if not pending_tasks:
        logger.info("没有待运行的任务")
        return

    # 尝试调度任务
    remaining_pending = []
    newly_running = []

    for task in pending_tasks:
        if task.gpu_count <= len(free_gpus):
            # 分配 GPU
            allocated = allocate_gpus(free_gpus, task.gpu_count)

            if allocated:
                # 启动任务
                pid = start_task(task, allocated)

                if pid:
                    task.assigned_gpus = allocated
                    task.pid = pid
                    task.start_time = datetime.now().isoformat()
                    newly_running.append(task)

                    # 更新空闲 GPU 列表
                    free_gpus = [g for g in free_gpus if g not in allocated]

                    logger.info(f"任务 {task.name} 已调度, GPUs: {allocated}")
                else:
                    remaining_pending.append(task)
            else:
                remaining_pending.append(task)
        else:
            remaining_pending.append(task)

    # 更新任务池
    save_pool(PENDING_FILE, remaining_pending)

    # 合并新运行的任务到 running 池
    running_tasks.extend(newly_running)
    save_pool(RUNNING_FILE, running_tasks)


def print_status() -> None:
    """打印当前状态"""
    pending = load_pool(PENDING_FILE)
    running = load_pool(RUNNING_FILE)
    completed = load_pool(COMPLETED_FILE)
    failed = load_pool(FAILED_FILE)

    logger.info(f"任务状态 - Pending: {len(pending)}, Running: {len(running)}, "
                f"Completed: {len(completed)}, Failed: {len(failed)}")

    if running:
        for task in running:
            logger.info(f"  运行中: {task.name} (GPUs: {task.assigned_gpus}, PID: {task.pid})")


# ============== 主循环 ==============

def main_loop() -> None:
    """主调度循环"""
    logger.info("=" * 50)
    logger.info("GPU 训练任务调度器启动")
    logger.info("=" * 50)

    # 初始化任务池文件
    for pool_file in [PENDING_FILE, RUNNING_FILE, COMPLETED_FILE, FAILED_FILE]:
        if not pool_file.exists():
            save_pool(pool_file, [])

    while True:
        try:
            logger.info("-" * 40)

            # 1. 同步配置文件中的新任务
            sync_pending_pool()

            # 2. 监控运行中的任务
            monitor_running_tasks()

            # 3. 调度新任务
            schedule_tasks()

            # 4. 打印状态
            print_status()

            # 5. 检查是否所有任务都已完成
            pending = load_pool(PENDING_FILE)
            running = load_pool(RUNNING_FILE)

            if not pending and not running:
                logger.info("所有任务已完成，调度器退出")
                break

            # 6. 等待下一次调度
            logger.info(f"等待 {SCHEDULE_INTERVAL} 秒后进行下一次调度...")
            time.sleep(SCHEDULE_INTERVAL)

        except KeyboardInterrupt:
            logger.info("收到中断信号，调度器退出")
            break
        except Exception as e:
            logger.error(f"调度循环出错: {e}")
            time.sleep(60)  # 出错后等待 1 分钟再重试


if __name__ == "__main__":
    main_loop()
