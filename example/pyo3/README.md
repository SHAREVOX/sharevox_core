# Python サンプルコード (PyO3 によるバインディング経由)

sharevox_core ライブラリ の Python バインディングを使った音声合成のサンプルコードです。

## 準備

TODO

- Python インタプリタ ≧3.8 + venv
- sharevox_core_python_api の whl (`pip install`)
- onnxruntime の DLL (/README.md と同様)
- open_jtalk_dic_utf_8-1.11 (/README.md と同様)

## 実行

Open JTalk 辞書ディレクトリ、読み上げさせたい文章、出力 wav ファイルのパスの 3 つを指定して run.py を実行します。

```console
❯ python ./run.py -h
usage: run.py [-h] [--mode MODE] open_jtalk_dict_dir text out

positional arguments:
  open_jtalk_dict_dir  Open JTalkの辞書ディレクトリ
  text                 読み上げさせたい文章
  out                  出力wavファイルのパス

optional arguments:
  -h, --help           show this help message and exit
  --mode MODE          モード ("AUTO", "CPU", "GPU")
```

```console
❯ # python ./run.py <Open JTalk辞書ディレクトリ> <読み上げさせたい文章> <出力wavファイルのパス>
❯ python ./run.py ./open_jtalk_dic_utf_8-1.11 これはテストです ./audio.wav
[DEBUG] run.py: sharevox_core.SUPPORTED_DEVICES=SupportedDevices(cpu=True, cuda=True, dml=False)
[INFO] run.py: Initializing (acceleration_mode=<AccelerationMode.AUTO: 'AUTO'>, open_jtalk_dict_dir=PosixPath('open_jtalk_dic_utf_8-1.11'))
[DEBUG] run.py: core.metas=[Meta(name='デモボイス', styles=[Style(name='ノーマル', id=1), Style(name='ノーマル', id=0)], speaker_uuid='3b000dd2-8ee2-4dae-8bc6-30dccb6087a3', version='0.0.2')]
[DEBUG] run.py: core.is_gpu_mode=True
[INFO] run.py: Loading model 0
[DEBUG] run.py: core.is_model_loaded(0)=True
[INFO] run.py: Creating an AudioQuery from 'これはテストです'
[INFO] run.py: Synthesizing with {"accent_phrases": [{"moras": [{"text": "コ", "consonant": "k", "consonant_length": 0.06324971, "vowel": "o", "vowel_length": 0.027723162, "pitch": 4.705461}, {"text": "レ", "consonant": "r", "consonant_length": 0.054514598, "vowel": "e", "vowel_length": 0.030226612, "pitch": 4.841061}, {"text": "ワ", "consonant": "w", "consonant_length": 0.08183504, "vowel": "a", "vowel_length": 0.10486739, "pitch": 4.9399757}], "accent": 3, "pause_mora": null, "is_interrogative": false}, {"moras": [{"text": "テ", "consonant": "t", "consonant_length": 0.035247125, "vowel": "e", "vowel_length": 0.06039452, "pitch": 5.200297}, {"text": "ス", "consonant": "s", "consonant_length": 0.07048588, "vowel": "U", "vowel_length": 0.049617834, "pitch": 0.0}, {"text": "ト", "consonant": "t", "consonant_length": 0.03180605, "vowel": "o", "vowel_length": 0.06136502, "pitch": 4.870039}, {"text": "デ", "consonant": "d", "consonant_length": 0.033858374, "vowel": "e", "vowel_length": 0.08559787, "pitch": 4.4296713}, {"text": "ス", "consonant": "s", "consonant_length": 0.18445624, "vowel": "U", "vowel_length": 0.1268351, "pitch": 0.0}], "accent": 1, "pause_mora": null, "is_interrogative": false}], "speed_scale": 1.0, "pitch_scale": 0.0, "intonation_scale": 1.0, "volume_scale": 1.0, "pre_phoneme_length": 0.1, "post_phoneme_length": 0.1, "output_sampling_rate": 48000, "output_stereo": false, "kana": "コレワ'/テ'_ストデ_ス"}
[INFO] run.py: Wrote `audio.wav`
[DEBUG] lib.rs: Destructing a SharevoxCore
```

正常に実行されれば音声合成の結果である wav ファイルが生成されます。
この例の場合、`"これはテストです"`という読み上げの wav ファイルが audio.wav という名前で生成されます。
