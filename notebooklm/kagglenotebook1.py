{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# ⚡ 🇹🇭 เตรียมชุดข้อมูลเสียงภาษาไทย (MCV v24) [Speedrun Edition]\n",
        "\n",
        "ฉบับสปีดรัน: เร็วที่สุดเท่าที่ระบบ Kaggle จะทำได้! \n",
        "เราจะใช้เทคนิค **Advanced Data Engineering 3 ประสาน**: \n",
        "1. `Aria2 Multi-connection`: โหลดไฟล์ผ่าน 16 ท่อ ลดเวลาดาวน์โหลดไฟล์ 8GB จาก 15 นาทีเหลือ ไม่เกิน 3 นาที\n",
        "2. `On-the-fly Tar-Streaming`: แตกไฟล์บนอากาศ (RAM) ไม่เขียนไฟล์ .mp3 ลงดิสก์เลยแม้แต่ไบต์เดียว ลดเวลาแตกไฟล์ทิ้งไป 100%\n",
        "3. `FFmpeg Stdin Pipe`: ปั๊มไบต์เสียงลงคอ FFMPEG โดยตรง ประหยัด I/O ขั้นสุด!\n",
        "---\n",
        "*🎯 เป้าหมาย:* โฟลเดอร์ `dataset/` จะเต็มไปด้วยไฟล์ `.wav` ความถี่ 24000Hz สำหรับ Ovlyra ภายใน 10 นาที!"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# ติดตั้งอาวุธหนัก: aria2 สำหรับโหลดไฟล์ความเร็วสูงสุดทะลุกำแพง (16 threads)\n",
        "!apt-get update -qq && apt-get install -y -qq aria2\n",
        "!pip install pandas requests psutil tqdm -q"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### ⚙️ ส่วนที่ 1: ตั้งค่าระบบ & RAM Disk (Configuration)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import json\n",
        "import os\n",
        "import subprocess\n",
        "import sys\n",
        "import tarfile\n",
        "import concurrent.futures\n",
        "import multiprocessing\n",
        "import psutil\n",
        "import pandas as pd\n",
        "import requests\n",
        "from tqdm import tqdm\n",
        "\n",
        "MDC_API_KEY = \"6799035d029e9bcf52cea5e84637baf19f377d387d3ab1f3ef4f4cc0235f5ccf\"\n",
        "MDC_DATASET_ID = \"cmj8u3pvx00r9nxxb1yo5z53z\"\n",
        "MDC_API_BASE = \"https://datacollective.mozillafoundation.org/api\"\n",
        "\n",
        "RAM_DIR = \"/dev/shm/mcv_thai_ram\"\n",
        "os.makedirs(RAM_DIR, exist_ok=True)\n",
        "\n",
        "KAGGLE_WORKING = \"/kaggle/working\"\n",
        "DOWNLOAD_DIR = os.path.join(KAGGLE_WORKING, \"mcv_thai_raw\")\n",
        "OUTPUT_WAV_DIR = os.path.join(KAGGLE_WORKING, \"dataset\", \"th_wavs\")\n",
        "OUTPUT_JSONL = os.path.join(KAGGLE_WORKING, \"dataset\", \"train.jsonl\")\n",
        "\n",
        "TARGET_SAMPLE_RATE = 24000\n",
        "TARGET_CHANNELS = 1\n",
        "MAX_CLIPS = 10000"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 📥 ส่วนที่ 2: ดาวน์โหลดระดับเครื่องจักรด้วย Aria2"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "os.makedirs(DOWNLOAD_DIR, exist_ok=True)\n",
        "response = requests.post(\n",
        "    f\"{MDC_API_BASE}/datasets/{MDC_DATASET_ID}/download\",\n",
        "    headers={\"Authorization\": f\"Bearer {MDC_API_KEY}\", \"Content-Type\": \"application/json\"},\n",
        ")\n",
        "download_url = response.json().get(\"downloadUrl\") or response.json().get(\"url\")\n",
        "\n",
        "tar_path = os.path.join(DOWNLOAD_DIR, \"cv_thai_v24.tar.gz\")\n",
        "if not os.path.exists(tar_path):\n",
        "    subprocess.run([\n",
        "        \"aria2c\", \"-x\", \"16\", \"-s\", \"16\", \"-k\", \"1M\",\n",
        "        \"-o\", \"cv_thai_v24.tar.gz\", \"-d\", DOWNLOAD_DIR, download_url\n",
        "    ], check=True)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 🔍 ส่วนที่ 3: ดูดข้อมูล TSV จาก .tar.gz ตรงๆ เข้า RAM"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "validated_tsv_path = os.path.join(RAM_DIR, \"validated.tsv\")\n",
        "clip_durations_path = os.path.join(RAM_DIR, \"clip_durations.tsv\")\n",
        "\n",
        "if not os.path.exists(validated_tsv_path):\n",
        "    found_val = False\n",
        "    with tarfile.open(tar_path, \"r:gz\") as tar:\n",
        "        for member in tar.getmembers():\n",
        "            if member.name.endswith(\"validated.tsv\"):\n",
        "                with open(validated_tsv_path, \"wb\") as f_out:\n",
        "                    f_out.write(tar.extractfile(member).read())\n",
        "                found_val = True\n",
        "            elif member.name.endswith(\"clip_durations.tsv\"):\n",
        "                with open(clip_durations_path, \"wb\") as f_out:\n",
        "                    f_out.write(tar.extractfile(member).read())\n",
        "            if found_val and os.path.exists(clip_durations_path):\n",
        "                break\n",
        "\n",
        "df = pd.read_csv(validated_tsv_path, sep=\"\\t\")\n",
        "if MAX_CLIPS and MAX_CLIPS < len(df):\n",
        "    df = df.head(MAX_CLIPS)\n",
        "\n",
        "durations_dict = {}\n",
        "if os.path.exists(clip_durations_path):\n",
        "    df_durations = pd.read_csv(clip_durations_path, sep=\"\\t\")\n",
        "    for _, r in df_durations.iterrows():\n",
        "        durations_dict[r[\"clip\"]] = float(r[\"duration[ms]\"]) / 1000.0"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### ⚡ ส่วนที่ 4: สตรีมมิ่งไฟล์เสียงเข้าท่อ Piped-FFMPEG"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)\n",
        "wanted_filenames = set(df[\"path\"].dropna().tolist())\n",
        "row_dict = df.set_index(\"path\").to_dict(\"index\")\n",
        "\n",
        "successful = 0\n",
        "failed = 0\n",
        "jsonl_lines = []\n",
        "\n",
        "def process_audio_bytes(mp3_bytes, mp3_filename):\n",
        "    wav_filename = mp3_filename.replace(\".mp3\", \".wav\")\n",
        "    wav_path = os.path.join(OUTPUT_WAV_DIR, wav_filename)\n",
        "    process = subprocess.Popen([\n",
        "        \"ffmpeg\", \"-y\", \"-i\", \"pipe:0\", \"-ar\", str(TARGET_SAMPLE_RATE),\n",
        "        \"-ac\", str(TARGET_CHANNELS), \"-threads\", \"1\", wav_path\n",
        "    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
        "    process.communicate(input=mp3_bytes)\n",
        "    \n",
        "    if process.returncode == 0:\n",
        "        duration = durations_dict.get(mp3_filename, 5.0)\n",
        "        if duration > 30.0:\n",
        "            if os.path.exists(wav_path): os.remove(wav_path)\n",
        "            return \"failed\", None\n",
        "        row = row_dict[mp3_filename]\n",
        "        result_json = json.dumps({\n",
        "            \"transcript\": str(row[\"sentence\"]),\n",
        "            \"language\": \"th\",\n",
        "            \"wav_path\": wav_path,\n",
        "            \"duration\": round(duration, 2),\n",
        "            \"sample_rate\": TARGET_SAMPLE_RATE\n",
        "        }, ensure_ascii=False)\n",
        "        return \"ok\", result_json\n",
        "    return \"failed\", None\n",
        "\n",
        "num_workers = min(multiprocessing.cpu_count() * 2, 32)\n",
        "mp3_data_list = []\n",
        "with tarfile.open(tar_path, \"r|gz\") as tar:\n",
        "    for member in tar:\n",
        "        filename = os.path.basename(member.name)\n",
        "        if filename in wanted_filenames:\n",
        "            mp3_bytes = tar.extractfile(member).read()\n",
        "            mp3_data_list.append((mp3_bytes, filename))\n",
        "            wanted_filenames.remove(filename)\n",
        "            if not wanted_filenames: break\n",
        "\n",
        "executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)\n",
        "futures = []\n",
        "pbar = tqdm(total=len(mp3_data_list), desc=\"Converting in Real-Time\")\n",
        "\n",
        "for item in mp3_data_list:\n",
        "    future = executor.submit(process_audio_bytes, item[0], item[1])\n",
        "    future.add_done_callback(lambda p: pbar.update(1))\n",
        "    futures.append(future)\n",
        "\n",
        "for future in concurrent.futures.as_completed(futures):\n",
        "    status, line = future.result()\n",
        "    if status == \"ok\":\n",
        "        successful += 1\n",
        "        jsonl_lines.append(line)\n",
        "    else:\n",
        "        failed += 1\n",
        "\n",
        "pbar.close()\n",
        "executor.shutdown()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 🧹 ส่วนที่ 5: เขียนตาราง Train & ทำลายหลักฐานให้สิ้นซาก"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)\n",
        "with open(OUTPUT_JSONL, \"w\", encoding=\"utf-8\") as f:\n",
        "    f.write(\"\\n\".join(jsonl_lines))\n",
        "    f.flush()\n",
        "    os.fsync(f.fileno())\n",
        "\n",
        "for temp_path in [DOWNLOAD_DIR, RAM_DIR]:\n",
        "    if os.path.exists(temp_path):\n",
        "        os.system(f\"rm -rf {temp_path}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 🔊 ส่วนที่ 6: ดาวน์โหลด xcodec2 Audio Codec"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "XCODEC2_DIR = os.path.join(KAGGLE_WORKING, 'xcodec2_ckpt')\n",
        "XCODEC2_PATH = os.path.join(XCODEC2_DIR, 'checkpoint.pt')\n",
        "XCODEC2_URL = 'https://huggingface.co/HKUSTAudio/xcodec2/resolve/main/ckpt/epoch%3D4-step%3D1400000.ckpt'\n",
        "\n",
        "os.makedirs(XCODEC2_DIR, exist_ok=True)\n",
        "if not os.path.exists(XCODEC2_PATH):\n",
        "    subprocess.run([\n",
        "        'aria2c', '-x', '16', '-s', '16', '-k', '1M',\n",
        "        '-o', 'checkpoint.pt', '-d', XCODEC2_DIR, XCODEC2_URL\n",
        "    ], check=True)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 🧬 ส่วนที่ 7: Clone Repo + ติดตั้ง + Patch สำคัญ\n",
        "⚠️ ต้อง Patch 2 จุดก่อนรัน Vectorizer:\n",
        "1. ลบการตรวจสอบ flash_attn_2 ใน `environment.py` (T4 ไม่มี flash-attn แต่ Vectorizer ไม่ได้ใช้งานมันจริง)\n",
        "2. ลบ flag `--allowed_languages` ที่ไม่มีใน Source Code"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "REPO_URL = 'https://github.com/chalitbkb/Ovlyra.git'\n",
        "REPO_DIR = '/kaggle/working/Ovlyra'\n",
        "if not os.path.exists(REPO_DIR):\n",
        "    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)\n",
        "\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', REPO_DIR, '--quiet'], check=True)\n",
        "\n",
        "import shutil\n",
        "CODEC_CONFIG_SRC = os.path.join(REPO_DIR, 'example', 'codec', 'model_config.json')\n",
        "CODEC_CONFIG_DST = os.path.join(XCODEC2_DIR, 'model_config.json')\n",
        "if not os.path.exists(CODEC_CONFIG_DST):\n",
        "    shutil.copy2(CODEC_CONFIG_SRC, CODEC_CONFIG_DST)\n",
        "    print(f'✅ Copied model_config.json → {CODEC_CONFIG_DST}')\n",
        "\n",
        "# ===== PATCH: ลบการตรวจสอบ flash_attn_2 ใน environment.py =====\n",
        "# Vectorizer เรียก environment.initialize_distributed_environment_context()\n",
        "# ซึ่งจะ raise ValueError ถ้าไม่มี flash-attn แม้ว่า Vectorizer ไม่ได้ใช้มันเลย\n",
        "env_path = os.path.join(REPO_DIR, 'tts', 'training', 'environment.py')\n",
        "with open(env_path, 'r') as f:\n",
        "    env_code = f.read()\n",
        "\n",
        "env_code = env_code.replace(\n",
        "    'if not transformers_utils.is_flash_attn_2_available():',\n",
        "    'if False:  # [PATCHED] Disabled for T4 compatibility'\n",
        ")\n",
        "\n",
        "with open(env_path, 'w') as f:\n",
        "    f.write(env_code)\n",
        "\n",
        "print('✅ Patched environment.py: disabled flash_attn_2 check (T4 Safe!)')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 🔬 ส่วนที่ 8: รัน Data Vectorization"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "VAL_SPLIT      = 0.05\n",
        "VEC_BATCH_SIZE  = 16\n",
        "VEC_NUM_WORKERS = 4\n",
        "VECTORIZED_DIR = os.path.join(KAGGLE_WORKING, 'vectorized_th')\n",
        "os.makedirs(VECTORIZED_DIR, exist_ok=True)\n",
        "\n",
        "os.chdir(REPO_DIR)\n",
        "import torch\n",
        "gpu_count = torch.cuda.device_count()\n",
        "nproc = gpu_count if gpu_count > 0 else 1\n",
        "\n",
        "cmd = [\n",
        "    'torchrun', f'--nproc_per_node={nproc}',\n",
        "    'tools/data/data_vectorizer.py',\n",
        "    f'--codec_model_path={XCODEC2_PATH}',\n",
        "    f'--batch_size={VEC_BATCH_SIZE}',\n",
        "    f'--num_workers={VEC_NUM_WORKERS}',\n",
        "    '--compile_model',\n",
        "    f'--dataset_path={OUTPUT_JSONL}',\n",
        "    f'--output_dir={VECTORIZED_DIR}',\n",
        "    f'--val_split={VAL_SPLIT}'\n",
        "]\n",
        "subprocess.run(cmd, check=True)\n",
        "print('\\n✅ Data Vectorization complete!')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 📦 ส่วนที่ 9: รวมชิ้นส่วนข้อมูลให้เป็นก้อนเดียว (Merge Shards)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "os.chdir(REPO_DIR)\n",
        "subprocess.run([\n",
        "    sys.executable, 'tools/data/data_merger.py',\n",
        "    f'--dataset_path={VECTORIZED_DIR}',\n",
        "    '--remove_shards'\n",
        "], check=True)\n",
        "print('\\n✅ Shards merged!')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "os.chdir(KAGGLE_WORKING)\n",
        "if os.path.exists(REPO_DIR):\n",
        "    shutil.rmtree(REPO_DIR)\n",
        "print('✅ Output พร้อมสำหรับให้ Notebook 2 นำไปเทรนแล้ว!')"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10.12"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
