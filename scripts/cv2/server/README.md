# UtterTune TTS API Server

CosyVoice2 + LoRAアダプター対応のText-to-Speech APIサーバー。

## 概要

UtterTune TTS APIは、CosyVoice2をベースにしたゼロショット音声合成サーバーです。LoRAアダプターによるファインチューニング済みモデルをサポートし、音韻制御タグ（`<PHON_START>...<PHON_END>`）による発音指定が可能です。

### 主な機能

- ゼロショット音声クローニング（プロンプト音声から声質を模倣）
- LoRAアダプターによるファインチューニング対応
- 音韻制御タグによる発音指定
- HTTP REST API + WebSocketリアルタイム合成
- ストリーミングレスポンス対応

## セットアップ

### 依存関係のインストール

```bash
cd /path/to/UtterTune
pip install -r requirements.txt
```

### サーバー起動

```bash
# ベースモデルのみ
python -m scripts.cv2.server.main \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --port 50000

# LoRAアダプター付き
python -m scripts.cv2.server.main \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \
    --port 50000

# GPU使用・FP16モード
python -m scripts.cv2.server.main \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \
    --fp16 \
    --port 50000

# CPUモード（デバッグ用）
python -m scripts.cv2.server.main \
    --base_model pretrained_models/CosyVoice2-0.5B \
    --cpu \
    --port 50000
```

### 環境変数による設定

`UTTERTUNE_` プレフィックスで環境変数を設定できます。

```bash
export UTTERTUNE_BASE_MODEL=pretrained_models/CosyVoice2-0.5B
export UTTERTUNE_LORA_DIR=lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS
export UTTERTUNE_PORT=8080
export UTTERTUNE_FP16=true
export UTTERTUNE_MAX_CONCURRENT_REQUESTS=4

python -m scripts.cv2.server.main
```

## API エンドポイント

### ヘルスチェック

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/health/` | GET | サービス稼働確認 |
| `/health/ready` | GET | モデルロード完了確認（Kubernetes Readiness Probe用） |
| `/health/model/info` | GET | モデル詳細情報取得 |

### TTS合成

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/v1/tts/` | POST | 音声合成（HTTP） |
| `/v1/tts/ws` | WebSocket | リアルタイム音声合成 |

## 使用例

### curl

```bash
# ヘルスチェック
curl http://localhost:50000/health/

# モデル情報取得
curl http://localhost:50000/health/model/info

# 音声合成（WAVファイル出力）
curl -X POST "http://localhost:50000/v1/tts/?format=wav" \
    -F "tts_text=こんにちは、世界" \
    -F "prompt_text=メロスは激怒した。" \
    -F "prompt_wav=@prompts/wav/sample.wav" \
    -o output.wav

# 音声合成（速度調整）
curl -X POST "http://localhost:50000/v1/tts/?format=wav&speed=1.2" \
    -F "tts_text=少し速く話します。" \
    -F "prompt_text=サンプルテキスト" \
    -F "prompt_wav=@prompt.wav" \
    -o output_fast.wav

# ストリーミング再生（PCM形式）
curl -X POST "http://localhost:50000/v1/tts/?stream=true&format=pcm" \
    -F "tts_text=ストリーミングで再生します。" \
    -F "prompt_text=サンプルテキスト" \
    -F "prompt_wav=@prompt.wav" \
    --output - | play -t raw -r 22050 -e signed -b 16 -c 1 -

# 音韻制御タグ使用（LoRA必須）
curl -X POST "http://localhost:50000/v1/tts/?format=wav" \
    -F "tts_text=<PHON_START>チ'ミ/モーリョー<PHON_END>が跋扈する。" \
    -F "prompt_text=サンプルテキスト" \
    -F "prompt_wav=@prompt.wav" \
    -o output_phon.wav
```

### Python (requests)

```python
import requests

# 基本的な音声合成
with open("prompt.wav", "rb") as f:
    response = requests.post(
        "http://localhost:50000/v1/tts/",
        files={"prompt_wav": f},
        data={
            "tts_text": "こんにちは、世界",
            "prompt_text": "プロンプトの書き起こし",
        },
        params={"format": "wav", "speed": 1.0},
    )

with open("output.wav", "wb") as f:
    f.write(response.content)

# ストリーミング受信
with open("prompt.wav", "rb") as f:
    response = requests.post(
        "http://localhost:50000/v1/tts/",
        files={"prompt_wav": f},
        data={
            "tts_text": "ストリーミングテスト",
            "prompt_text": "プロンプトテキスト",
        },
        params={"stream": "true", "format": "pcm"},
        stream=True,
    )

# チャンクごとに処理
for chunk in response.iter_content(chunk_size=4096):
    process_audio_chunk(chunk)  # リアルタイム再生など
```

### Python (WebSocket)

```python
import asyncio
import json
import numpy as np
import websockets

async def tts_websocket_client():
    uri = "ws://localhost:50000/v1/tts/ws"

    async with websockets.connect(uri) as ws:
        # 1. プロンプト音声を送信（16kHz, mono, int16 PCM）
        prompt_audio = load_audio_as_int16("prompt.wav", sample_rate=16000)
        await ws.send(prompt_audio.tobytes())

        # 確認メッセージを受信
        response = await ws.recv()
        print(json.loads(response))  # {"status": "prompt_received", ...}

        # 2. 設定を送信
        config = {
            "prompt_text": "プロンプトの書き起こし",
            "speed": 1.0,
        }
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
        save_audio(full_audio, "output.wav", sample_rate=22050)

asyncio.run(tts_websocket_client())
```

### JavaScript (WebSocket)

```javascript
const ws = new WebSocket("ws://localhost:50000/v1/tts/ws");
ws.binaryType = "arraybuffer";

const audioContext = new AudioContext({ sampleRate: 22050 });
let audioQueue = [];

ws.onopen = async () => {
    // 1. プロンプト音声を送信（16kHz, mono, int16 PCM）
    const promptAudio = await loadAudioAsInt16Array("prompt.wav", 16000);
    ws.send(promptAudio.buffer);
};

ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
        // 音声チャンク受信
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

// 音声をInt16配列として読み込む関数
async function loadAudioAsInt16Array(url, targetSampleRate) {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const audioCtx = new AudioContext({ sampleRate: targetSampleRate });
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const float32 = audioBuffer.getChannelData(0);
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32767));
    }
    return int16;
}
```

## API リファレンス

### POST /v1/tts/

音声を合成します。

#### リクエスト（multipart/form-data）

| パラメータ | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| `tts_text` | string | Yes | 合成するテキスト |
| `prompt_text` | string | Yes | プロンプト音声の書き起こし |
| `prompt_wav` | file | Yes | プロンプト音声ファイル（4秒以下推奨） |

#### クエリパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `speed` | float | 1.0 | 話速（0.5〜2.0） |
| `stream` | bool | false | ストリーミングレスポンス |
| `format` | string | "wav" | 出力形式（"wav" or "pcm"） |

#### レスポンス

- **WAV形式**: `audio/wav`（Content-Type）
- **PCM形式**: `audio/pcm`（22050Hz, mono, int16）
  - ヘッダー: `X-Sample-Rate: 22050`, `X-Bit-Depth: 16`, `X-Channels: 1`

#### エラーレスポンス

| ステータス | 説明 |
|-----------|------|
| 400 | 不正なリクエスト（stream=true かつ format=wav など） |
| 500 | 合成エラー |
| 503 | モデル未ロード |

### WebSocket /v1/tts/ws

リアルタイム音声合成用WebSocketエンドポイント。

#### プロトコル

1. **クライアント → サーバー（バイナリ）**: プロンプト音声
   - 形式: 16kHz, mono, int16 PCM（生バイト列）
   - 最大長: 10秒（160,000サンプル）

2. **クライアント → サーバー（JSON）**: 設定・合成リクエスト
   ```json
   // 設定更新
   {"prompt_text": "書き起こし", "speed": 1.0}

   // 合成リクエスト
   {"text": "合成するテキスト"}
   ```

3. **サーバー → クライアント（バイナリ）**: 合成音声
   - 形式: 22050Hz, mono, int16 PCM

4. **サーバー → クライアント（JSON）**: ステータス
   ```json
   {"status": "prompt_received", "samples": 48000, "duration_sec": 3.0}
   {"status": "config_updated"}
   {"status": "done"}
   {"error": "エラーメッセージ"}
   ```

## 設定オプション

| オプション | 環境変数 | デフォルト | 説明 |
|-----------|----------|----------|------|
| `--base_model` | `UTTERTUNE_BASE_MODEL` | - | CosyVoice2ベースモデルパス |
| `--lora_dir` | `UTTERTUNE_LORA_DIR` | None | LoRAアダプターパス |
| `--host` | `UTTERTUNE_HOST` | 0.0.0.0 | バインドアドレス |
| `--port` | `UTTERTUNE_PORT` | 50000 | ポート番号 |
| `--fp16` | `UTTERTUNE_FP16` | false | FP16推論 |
| `--cpu` | `UTTERTUNE_USE_CPU` | false | CPU強制使用 |
| `--seed` | `UTTERTUNE_SEED` | 42 | 乱数シード |
| - | `UTTERTUNE_MAX_CONCURRENT_REQUESTS` | 2 | 最大同時リクエスト数 |

## 音声仕様

| 項目 | 入力（プロンプト） | 出力（合成音声） |
|------|------------------|-----------------|
| サンプルレート | 16,000 Hz | 22,050 Hz |
| チャンネル | モノラル | モノラル |
| ビット深度 | 16-bit (int16) | 16-bit (int16) |
| 推奨長さ | 4秒以下 | - |
| 最大長さ（WS） | 10秒 | - |

## パフォーマンス

| 環境 | 推定処理時間 |
|------|-------------|
| GPU (CUDA) | 0.5〜2秒/文 |
| CPU | 5〜20秒/文 |

- GPU VRAM: 約2〜4GB（ベースモデル）
- 同時リクエスト数は `max_concurrent_requests` で制限（デフォルト: 2）

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

# またはCPUモードで実行
python -m scripts.cv2.server.main --cpu
```

### 音声が歪む・速度がおかしい

- プロンプト音声が16kHzか確認
- WebSocket使用時は必ず16kHz mono int16 PCMで送信

## ライセンス

UtterTuneプロジェクトのライセンスに従います。
