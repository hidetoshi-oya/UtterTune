# UtterTune

**G2Pを使用しないLLMベースTTSのための、LoRAによる音素レベルの発音・韻律制御** (現在は **[CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice)** の**日本語**をサポート)

[![arXiv](https://img.shields.io/badge/arXiv-2508.09767-b31b1b.svg)](https://www.arxiv.org/abs/2508.09767)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS)
[![Static Demo](https://img.shields.io/badge/Demo-GitHub%20Pages-blue)](https://shuheikatoinfo.github.io/UtterTune)

## 概要

UtterTuneは、明示的なG2P（Grapheme-to-Phoneme）を持たないLLMベースTTSにおいて、**音素レベルの発音と韻律を編集・制御**するための軽量LoRAアダプターおよびツールセットです。特殊タグトークン（`<PHON_START>`, `<PHON_END>`）を使用することで、表音文字（日本語の場合はかな）を用いてモデルに正しい発音を教えることができます。

**モデルサイズ**: 10MB未満（LoRAアダプター）。元のCosyVoice 2-0.5Bモデルは約1GBです。

## 特徴

- **LoRAファインチューニング**: ベースモデルのLLMコンポーネントのフルファインチューニングは不要
- **特殊トークン挿入**: `<PHON_START>`, `<PHON_END>`による音素レベルの発音制御
- **他言語への影響なし**: LoRAは言語固有
- **学習済み重みを公開**: [Hugging Faceからダウンロード可能](https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS)
- **REST APIサーバー**: HTTP + WebSocketによるストリーミング対応
- **リアルタイムストリーミング再生**: 生成しながら音声を再生

## クイックスタート

### 1. リポジトリのクローンとサブモジュールの更新

```bash
git clone https://github.com/your-username/UtterTune.git
cd UtterTune
git submodule update --init --recursive
```

### 2. 学習済みモデルのダウンロード

```bash
mkdir -p pretrained_models

# CosyVoice2-0.5Bのダウンロード
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B

# LoRA重みのダウンロード
git lfs install
git clone https://huggingface.co/shuheikatoinfo/UtterTune-CosyVoice2-ja-JSUTJVS lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS
```

### 3. 仮想環境のセットアップ

```bash
# venvの作成（Python 3.10推奨）
python -m venv .venv
. .venv/bin/activate

# CosyVoice依存関係のインストール
pip install -r submodules/CosyVoice/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# UtterTune依存関係のインストール
pip install -r requirements.txt

# CosyVoiceリポジトリへのパスを追加
python - <<'PY'
import site, os
sp = next(p for p in site.getsitepackages() if p.endswith("site-packages"))
pth = os.path.join(sp, "cosyvoice_submodule.pth")
with open(pth, "w", encoding="utf-8") as f:
    f.write(os.path.abspath("submodules/CosyVoice") + "\n")
    f.write(os.path.abspath("submodules/CosyVoice/third_party/Matcha-TTS") + "\n")
print("Wrote:", pth)
PY
```

**soxの互換性問題が発生した場合:**

```bash
# Ubuntu
sudo apt-get install sox libsox-dev
# CentOS
sudo yum install sox sox-devel
```

## 使い方

### バッチ推論

テキストから音声ファイルを生成:

```bash
python -m scripts.cv2.infer \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \
    --texts "魑魅魍魎が跋扈する。|<PHON_START>チ'ミ/モーリョー<PHON_END>が<PHON_START>バ'ッコ<PHON_END>する。" \
    --prompt_wav prompts/wav/sample.wav \
    --prompt_text "プロンプト音声の書き起こし" \
    --out_dir wavs_out
```

**オプション:**

| オプション | 説明 |
|-----------|------|
| `--base_model` | CosyVoice2ベースモデルのディレクトリ（必須） |
| `--lora_dir` | LoRAアダプターのディレクトリ |
| `--texts` | 合成するテキスト（`\|`区切り、またはテキストファイルパス） |
| `--prompt_wav` | プロンプト音声ファイル（4秒以下推奨） |
| `--prompt_text` | プロンプト音声の書き起こし |
| `--out_dir` | 出力ディレクトリ（デフォルト: `wavs_out`） |
| `--trim_out` | 合成音声をトリミング |
| `--cpu` | CPU推論を強制 |
| `--seed` | 乱数シード（デフォルト: 42） |

### ストリーミング再生

生成しながらリアルタイムで音声を再生:

```bash
python -m scripts.cv2.stream_play \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \
    --texts "こんにちは、世界" \
    --prompt_wav prompts/wav/sample.wav \
    --prompt_text "プロンプト音声の書き起こし" \
    --speed 1.0 \
    --save
```

**追加オプション:**

| オプション | 説明 |
|-----------|------|
| `--speed` | 話速（デフォルト: 1.0） |
| `--save` | 合成音声をファイルにも保存 |

### APIサーバー

HTTPとWebSocketに対応したREST APIサーバーを起動:

```bash
# 基本的な起動
python -m scripts.cv2.server.main \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \
    --port 50000

# FP16推論を有効化
python -m scripts.cv2.server.main \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \
    --fp16 \
    --port 50000
```

**環境変数による設定:**

```bash
export UTTERTUNE_BASE_MODEL=pretrained_models/CosyVoice2-0.5B
export UTTERTUNE_LORA_DIR=lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS
export UTTERTUNE_PORT=50000
export UTTERTUNE_FP16=true
export UTTERTUNE_MAX_CONCURRENT_REQUESTS=4
```

**推論最適化オプション:**

低レイテンシ・高スループットのために複数の最適化オプションを利用できます：

```bash
# JIT + TensorRT + FP16 で最大限の最適化
python -m scripts.cv2.server.main \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \
    --jit --trt --fp16 \
    --port 50000
```

| オプション | 環境変数 | 説明 |
|-----------|---------|------|
| `--fp16` | `UTTERTUNE_FP16` | FP16（半精度）推論。メモリ使用量削減と高速化 |
| `--jit` | `UTTERTUNE_LOAD_JIT` | JITコンパイル済みflow encoderを使用（`flow.encoder.*.zip`が必要） |
| `--trt` | `UTTERTUNE_LOAD_TRT` | TensorRT最適化flow decoderを使用（`*.plan`ファイルが必要） |

> **Note:** JITとTensorRTを使用するには、事前にコンパイル済みファイルを生成する必要があります。詳細はCosyVoice2のドキュメントを参照してください。

**APIエンドポイント:**

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/health/` | GET | ヘルスチェック |
| `/health/ready` | GET | モデルロード状態（Kubernetesレディプローブ用） |
| `/health/model/info` | GET | モデル情報 |
| `/v1/tts/` | POST | 音声合成 |
| `/v1/tts/ws` | WebSocket | リアルタイムストリーミング合成 |

**curlリクエスト例:**

```bash
curl -X POST "http://localhost:50000/v1/tts/?format=wav" \
    -F "tts_text=<PHON_START>チ'ミ/モーリョー<PHON_END>が跋扈する。" \
    -F "prompt_text=サンプルテキスト" \
    -F "prompt_wav=@prompts/wav/sample.wav" \
    -o output.wav
```

詳細なAPIドキュメントは [scripts/cv2/server/README.md](scripts/cv2/server/README.md) を参照してください。

### クライアントからのリクエスト

#### HTTPストリーミング

```bash
# 付属のPythonクライアントを使用
python -m scripts.cv2.server.client http \
    --server http://localhost:50000 \
    --prompt-wav prompts/wav/sample.wav \
    --prompt-text "プロンプトの書き起こし" \
    --text "こんにちは、世界" \
    --speed 1.0 \
    --save output.wav
```

#### WebSocketストリーミング

```bash
# 付属のPythonクライアントを使用
python -m scripts.cv2.server.client ws \
    --server ws://localhost:50000 \
    --prompt-wav prompts/wav/sample.wav \
    --prompt-text "プロンプトの書き起こし" \
    --text "こんにちは、世界" \
    --speed 1.0
```

#### WebSocket (Python)

```python
import asyncio
import json
import numpy as np
import websockets

async def tts_websocket():
    uri = "ws://localhost:50000/v1/tts/ws"

    async with websockets.connect(uri) as ws:
        # 1. プロンプト音声を送信（16kHz, mono, int16 PCM）
        prompt_audio = load_audio_as_int16("prompt.wav", sample_rate=16000)
        await ws.send(prompt_audio.tobytes())

        # 確認メッセージを受信
        response = await ws.recv()
        print(json.loads(response))  # {"status": "prompt_received", ...}

        # 2. 設定を送信
        config = {"prompt_text": "プロンプトの書き起こし", "speed": 1.0}
        await ws.send(json.dumps(config))
        response = await ws.recv()
        print(json.loads(response))  # {"status": "config_updated"}

        # 3. テキストを送信して音声を受信
        await ws.send(json.dumps({"text": "こんにちは、世界"}))

        audio_chunks = []
        while True:
            data = await ws.recv()

            if isinstance(data, bytes):
                # 音声チャンク（22050Hz, mono, int16 PCM）
                audio_chunks.append(np.frombuffer(data, dtype=np.int16))
            else:
                msg = json.loads(data)
                if msg.get("status") == "done":
                    break

        # 音声を結合
        full_audio = np.concatenate(audio_chunks)
        return full_audio

asyncio.run(tts_websocket())
```

#### WebSocket (JavaScript)

```javascript
const ws = new WebSocket("ws://localhost:50000/v1/tts/ws");
ws.binaryType = "arraybuffer";

const audioContext = new AudioContext({ sampleRate: 22050 });

ws.onopen = async () => {
    // 1. プロンプト音声を送信（16kHz, mono, int16 PCM）
    const promptAudio = await loadAudioAsInt16Array("prompt.wav", 16000);
    ws.send(promptAudio.buffer);
};

ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
        // 音声チャンク受信 → 再生
        const int16Array = new Int16Array(event.data);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32767;
        }
        playAudioChunk(float32Array);
    } else {
        // JSONステータス
        const msg = JSON.parse(event.data);
        console.log("Status:", msg);

        if (msg.status === "prompt_received") {
            // 設定送信
            ws.send(JSON.stringify({
                prompt_text: "プロンプトの書き起こし",
                speed: 1.0
            }));
        } else if (msg.status === "config_updated") {
            // テキスト送信
            ws.send(JSON.stringify({ text: "こんにちは、世界" }));
        }
    }
};
```

### LoRAマージとvLLMでの運用

高スループットが必要な本番環境では、LoRAをベースモデルにマージしてvLLMで運用できます。

#### 1. LoRAのマージ

LoRAアダプターをベースモデルにマージし、vLLM互換形式でエクスポート:

```bash
python -m scripts.cv2.merge_lora_for_vllm \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \
    --output_dir pretrained_models/CosyVoice2-0.5B/vllm_merged
```

**オプション:**

| オプション | 説明 |
|-----------|------|
| `--base_model` | CosyVoice2ベースモデルのディレクトリ（必須） |
| `--lora_dir` | LoRAアダプターのディレクトリ（必須） |
| `--output_dir` | 出力ディレクトリ（デフォルト: `{base_model}/vllm_merged`） |
| `--device` | 使用デバイス（デフォルト: `cuda`） |

マージ処理では以下が行われます:
- LoRA重みをベースモデルに統合
- 新規トークン（`<PHON_START>`, `<PHON_END>`）の埋め込みを統合
- vLLM互換の`Qwen2ForCausalLM`形式でエクスポート

#### 2. vLLMでの起動

```bash
# vLLMサーバーを起動
python -m vllm.entrypoints.openai.api_server \
    --model pretrained_models/CosyVoice2-0.5B/vllm_merged \
    --trust-remote-code \
    --dtype bfloat16 \
    --port 8000
```

**vLLMの利点:**
- 高スループット（バッチ処理、連続バッチング）
- 低レイテンシ（PagedAttention、KVキャッシュ最適化）
- マルチGPU対応（テンソル並列）

#### 3. UtterTuneサーバーでvLLMを使用

UtterTuneのAPIサーバーでもvLLMバックエンドを使用できます：

```bash
# LoRAマージ済みモデルでvLLMを使用
python -m scripts.cv2.server.main \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --vllm_model_dir pretrained_models/CosyVoice2-0.5B/vllm_merged \
    --port 50000

# vLLMのみ有効化（デフォルトのvllmディレクトリを使用）
python -m scripts.cv2.server.main \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --vllm \
    --port 50000
```

**環境変数による設定:**

```bash
export UTTERTUNE_BASE_MODEL=pretrained_models/CosyVoice2-0.5B
export UTTERTUNE_LOAD_VLLM=true
export UTTERTUNE_VLLM_MODEL_DIR=pretrained_models/CosyVoice2-0.5B/vllm_merged
export UTTERTUNE_PORT=50000
```

**vLLM関連オプション:**

| オプション | 説明 |
|-----------|------|
| `--vllm` | vLLMバックエンドを有効化 |
| `--vllm_model_dir` | マージ済みvLLMモデルのパス（指定時は`--vllm`が自動有効化） |

> **Note:** `--vllm_model_dir`を指定すると、LoRAをランタイムで適用する代わりに、マージ済みのモデルを直接読み込みます。これによりメモリ効率と推論速度が向上します。

## 学習

### 1. データ準備

[JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) と [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) コーパスをダウンロードし、単語の一部を `<PHON_START>` と `<PHON_END>` タグを使用して発音に置き換えます:

```yaml
# 元のテキスト
BASIC5000_0004: 一週間して、そのニュースは本当になった。

# 置き換え後
BASIC5000_0004: <PHON_START>イッシュ'ーカン<PHON_END>して、そのニュースは本当になった。
```

次に、マニフェストを準備:

```bash
python -m scripts.cv2.extract_speech_tokens ...
python -m scripts.cv2.prepare_manifest ...
```

### 2. 学習設定

`configs/train/jsutjvs.yaml` を編集:

```yaml
manifest: data/manifests/all.tsv
base_model: pretrained_models/CosyVoice2-0.5B
val_ratio: 0.05
lora:
  rank: 16
  alpha: 64
  dropout: 0.05
  bias: none
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
training:
  output_dir: experiments/cv2/jsutjvs
  per_device_train_batch_size: 8
  max_steps: 20000
  learning_rate: 1e-4
  fp16: true
```

### 3. 学習の実行

```bash
python -m scripts.cv2.train --config configs/train/jsutjvs.yaml
```

チェックポイントからの再開:

```bash
python -m scripts.cv2.train \
    --config configs/train/jsutjvs.yaml \
    --resume_from_checkpoint experiments/cv2/jsutjvs/checkpoint-XXXX
```

## プロジェクト構成

```
UtterTune/
├── configs/
│   └── train/
│       └── jsutjvs.yaml          # 学習設定
├── scripts/
│   └── cv2/
│       ├── infer.py              # バッチ推論
│       ├── stream_play.py        # ストリーミング再生
│       ├── train.py              # LoRA学習
│       ├── extract_speech_tokens.py
│       ├── prepare_manifest.py
│       ├── merge_lora_for_vllm.py
│       └── server/
│           ├── main.py           # APIサーバーエントリーポイント
│           ├── config.py         # サーバー設定
│           ├── models.py         # Pydanticモデル
│           ├── dependencies.py   # FastAPI依存関係
│           ├── client.py         # Pythonクライアント
│           └── routers/
│               ├── health.py     # ヘルスチェックエンドポイント
│               └── tts.py        # TTSエンドポイント
├── pretrained_models/            # ベースモデル（git clone）
├── lora_weights/                 # LoRAアダプター
├── prompts/
│   └── wav/                      # プロンプト音声ファイル
├── submodules/
│   └── CosyVoice/                # CosyVoice2サブモジュール
└── requirements.txt
```

## 音声仕様

| 項目 | 入力（プロンプト） | 出力（合成音声） |
|------|------------------|-----------------|
| サンプルレート | 16,000 Hz | 22,050 Hz |
| チャンネル | モノラル | モノラル |
| ビット深度 | 16-bit (int16) | 16-bit (int16) |
| 推奨長さ | 2〜4秒 | - |

### プロンプト音声（prompt_wav）の要件

音声クローニングの再現性を高めるため、以下の要件を満たすプロンプト音声を使用してください。

#### 必須要件

| 項目 | 要件 |
|------|------|
| サンプルレート | **16,000 Hz**（他のレートは内部で変換されるが、16kHzが最適） |
| チャンネル | **モノラル**（ステレオは平均化される） |
| 長さ | **2〜4秒**（短すぎると特徴抽出が不十分、長すぎると品質低下） |
| フォーマット | WAV (PCM 16-bit) |

#### 品質要件

- **クリアな音声**: 背景ノイズ、残響、BGMがないこと
- **自然な発話**: 極端に感情的でない、平坦すぎない自然なイントネーション
- **完全な文**: 途中で切れていない、一文を最後まで発話した音声
- **適切な音量**: クリッピングがなく、十分な音量があること

#### prompt_textとの一致

`prompt_text`はプロンプト音声の内容と**完全に一致**させてください。不一致があると話者埋め込みの抽出精度が低下し、クローニング品質に影響します。

#### 前処理コマンド例

既存の音声ファイルを推奨フォーマットに変換:

```bash
# 基本的な変換（16kHz、モノラル、4秒以内）
ffmpeg -i input.wav -ar 16000 -ac 1 -t 4 output_16k_mono.wav

# 音量正規化 + 無音除去付き
ffmpeg -i input.wav \
  -ar 16000 \
  -ac 1 \
  -af "loudnorm=I=-16:TP=-1.5:LRA=11,silenceremove=1:0:-50dB:1:0:-50dB" \
  -t 4 \
  output_normalized.wav
```

#### 良いプロンプト音声の例

```
prompts/wav/common_voice_ja_41758953.wav  # 16kHz, モノラル, 2.9秒 ✅
prompts/wav/common_voice_ja_36360364.wav  # 16kHz, モノラル, 3.4秒 ✅
```

#### 避けるべきプロンプト音声

- 長すぎる音声（10秒以上）
- 複数人の声が含まれる音声
- 音楽やSEが含まれる音声
- 電話越しなど低品質な録音
- ささやき声や叫び声

## 必要要件

コア依存関係:

- Python 3.10以上
- PyTorch
- PEFT (LoRA)
- Transformers

サーバー依存関係:

- FastAPI
- Uvicorn
- WebSockets

詳細は `requirements.txt` を参照してください。

## トラブルシューティング

### モデルが読み込めない

```bash
# パスを確認
ls -la pretrained_models/CosyVoice2-0.5B/
ls -la lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS/
```

### CUDA out of memory

```bash
# 同時リクエスト数を減らす
export UTTERTUNE_MAX_CONCURRENT_REQUESTS=1

# またはCPUモードを使用
python -m scripts.cv2.server.main --cpu
```

### 音声品質の問題

- プロンプト音声が16kHzモノラルであることを確認
- プロンプト音声は4秒以下に
- WebSocket使用時は16kHzモノラルint16 PCMで送信

## 引用

研究でUtterTuneを使用する場合は、[論文](https://www.arxiv.org/abs/2508.09767)を引用してください:

```bibtex
@misc{Kato2025UtterTune,
  title={UtterTune: LoRA-Based Target-Language Pronunciation Edit and Control in Multilingual Text-to-Speech},
  author={Kato, Shuhei},
  year={2025},
  howpublished={arXiv:2508.09767 [cs.CL]},
}
```

## ライセンス

詳細は [LICENSE](LICENSE) を参照してください。学習済みLoRA重みは、学習データの制約により**非商用ライセンス**です。
