## 1. Detekcja za pomocą modelu mamba-yolo dotrenowanego na zbiorze AFO

```bash
python tools/gen_det_afo_mamba.py \
    --split test \
    --data_root ./data/afo \
    --model_path ./weights/best_two.pt \
    --save_dir ./out/det_results_mamba/afo/{split} \
    --generate_meta_data \
    --vis \
    --imgsz 960 \
    --conf 0.05
```

## 2. Ewaluacja wyników detekcji (bez uwzględniania wyników klasyfikacji - tylko jakość samego dopasowanie bounding boxów)

```bash
/home/iwo/Code/rl-mamba/io-mamba-rl/.venv/bin/python /home/iwo/Code/rl-mamba/io-mamba-rl/tools/det_eval_afo.py >> ./out/eval_results/mamba_det_eval.txt
```

## 3. Uruchomienie śledzenia
```bash
python -m tools.track_kalman \
    --det_path out/det_results_mamba/afo/test \
    --motion byte \
    --data_root data/afo/{split}/img/{seq}_{frame_id:03d}.jpg \
    --save_dir out/track_res_mamba/{dataset_name}/{split} \
    --vis
```