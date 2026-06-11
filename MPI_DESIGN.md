# GANSU MPI 化 設計ドキュメント (叩き台 v0.1)

2026-06-11 着手。直前セッションの方針メモ `project_steom_scaleout_policy.md` を実装計画に落とす。
**本ドキュメントは設計の叩き台であり、コードはまだ書かない**（user 方針: 大プロジェクト・計画的・blind-code 不可）。
ビルド/検証はリモート GPU マシン専用。

---

## 0. 目的とゴール

### なぜ MPI か（直前セッションの確定事項）
- RI Term A で **メモリ壁** (d_eri_vvvv 64.5GB/device) は解決済み、commit 済 (`81a7e39`)。
- 残るは **時間壁**: STEOM stage-2 build を GPU0 が逐次実行（IP build_dressed 53.7s + EA ~50s ≈ 104s）、
  その間 GPU1-7 が idle。= 大幅な速度犠牲 → scale-out が筋。
- **本命 = MPI/マルチプロセス**。理由:
  1. **マルチノードは multi-process が不可避**（1プロセスは複数ノードを跨げない）。標準 = MPI。
  2. **device-0 hardcode 問題が"消える"**: 各 rank を 1 GPU に pin すれば、その rank にとって自GPUが device 0。
     既存の `cudaSetDevice(0)` / native operator の device-0 前提が **rank ごと無改変で動く**
     → multi-thread 案で必要だった「operator device パラメータ化 refactor」が丸ごと不要。
  3. **single-node でも動く**: `mpirun -np N ./gansu` を1ノードで → N rank × 1 GPU。
     既存 A100×8 / H200×4 で開発・検証でき、**multi-node はコード変更ゼロで後から拡張**。

### ゴール（段階）
- **G1 (本ドキュメントの主眼)**: single-node multiprocess。`mpirun -np N` で N rank × 1 GPU、
  NCCL を `ncclCommInitRank` + MPI Bcast(uniqueId) で張る。既存 single-process 経路は無改変で残す
  （`-np 1` または非 MPI ビルドで現状と byte-identical）。
- **G2**: STEOM stage-2 を rank0=IP / rank1=EA 並列。bar-H を cross-rank 転送して STEOM を rank0 で。
- **G3**: multi-node（hostfile を渡すだけ、コード変更ゼロを目標）。

---

## 1. 現状アーキテクチャ（single-process / NCCL）

| 項目 | 実装 | 箇所 |
|---|---|---|
| GPU管理 | `MultiGpuManager` singleton が **1プロセスで** N device 分の cuBLAS/cuSOLVER handle・stream・NCCL comm を保持 | `include/multi_gpu_manager.hpp` |
| NCCL init | `ncclCommInitAll(comms, N, devlist)` — 1プロセス用 convenience API。uniqueId Bcast 不要 | `src/multi_gpu_manager.cu:58` |
| device 切替 | `DeviceGuard` (RAII) が `cudaSetDevice(d)` で全GPUを巡回 | `multi_gpu_manager.hpp:116` |
| collective | `nccl::all_reduce/broadcast/send/recv(..., device_id, stream)` — `device_id` で comm を選ぶ。**rank ≡ device index** | `include/nccl_comm.hpp` |
| init 呼出 | `rhf.cu` (×3), `dlpno_pair_data.cu` (×2), `dlpno_{ip,ea}_eom_native_operator.cu`, `thc_*.cu` | 8箇所 |
| ビルド flag | NCCL ライブラリ発見時に `target_compile_definitions(gansu PUBLIC GANSU_MULTI_GPU)` | `CMakeLists.txt:343-345` |
| device-0 前提 | `cudaSetDevice(0)` が 19 ファイルに散在（fit 判定・restore・native operator の基準デバイス） | 例 `eri_stored_ip_eom_ccsd.cu:346` |

**中核の概念モデル**: 「1プロセスが N GPU を所有し、`device_id` で巡回」。
NCCL の "rank" は **プロセス内の device index** に一致している。

---

## 2. 目標アーキテクチャ（MPI + NCCL ハイブリッド）

**標準構成**: MPI = プロセス/ノード管理・host 間通信、NCCL = ノード内 GPU collective。
GPU collective は引き続き NCCL（高速）、起動と uniqueId 配布とノード跨ぎ host 通信を MPI が担う。

```
mpirun -np N ./gansu ...
  rank 0  ── local_rank 0 ── CUDA_VISIBLE_DEVICES=0 ── GPU を device 0 と認識
  rank 1  ── local_rank 1 ── CUDA_VISIBLE_DEVICES=1 ── GPU を device 0 と認識
  ...
  rank N-1                                            ── GPU を device 0 と認識

  NCCL: rank0 が ncclGetUniqueId → MPI_Bcast(id) → 各 rank ncclCommInitRank(comm, N, id, rank)
        → 1プロセス1 comm、collective は (rank, world_size) で動作
```

### 2.1 rank ↔ device マッピング
- **local rank** を `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` で算出（ノード内順位）。
- 各 rank は **GPU を 1 枚だけ**使う。pin 方法は 2 案:
  - **(A) `CUDA_VISIBLE_DEVICES` をランチャで設定**（最もクリーン）。各 rank が自GPUを device 0 と見る
    → device-0 hardcode が無改変。ランチャ script で `CUDA_VISIBLE_DEVICES=$LOCAL_RANK` を export。
  - **(B) プロセス内で `cudaSetDevice(local_rank)`**。この場合 device-0 hardcode は壊れる（その rank の
    自GPUは device `local_rank`）→ 不採用。
- **採用 = (A)**。直前メモの核心（device-0 hardcode を無改変で活かす）が (A) でのみ成立。
  → ランチャは `mpirun -np N --bind-to none ./gansu_launch.sh ...`、`gansu_launch.sh` 内で
    `export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec ./gansu "$@"`。
  - これにより **C++ 側の `MultiGpuManager` は "1 device しか見えない" = num_devices_==1 と等価に縮退**でき、
    既存 single-GPU 経路がほぼそのまま各 rank で走る。

### 2.2 NCCL init の置換
`ncclCommInitAll`（1プロセス N comm）→ `ncclCommInitRank`（1プロセス 1 comm）:
```cpp
ncclUniqueId id;
if (world_rank == 0) ncclGetUniqueId(&id);
MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
ncclCommInitRank(&comm_, world_size, id, world_rank);   // この rank の唯一の comm
```
`MultiGpuManager` は MPI モードでは **comm を 1 個だけ**持つ（`nccl_comm(0)` が自 rank の comm）。

### 2.3 collective API の意味変更
- 現 `nccl::all_reduce(..., device_id, stream)`: `device_id` は **プロセス内 device index**。
- MPI モード: 各 rank は 1 comm のみ → `device_id` は常に 0（= 自 rank の comm）。
  collective は world 全 rank に跨って働く（`ncclCommInitRank` の world_size で決まる）。
- **API シグネチャは変えず**、`device_id=0` 固定で呼ぶ運用に。`send/recv` の `peer` は **world rank**。

---

## 3. 影響範囲（移行が必要なコード）

| 層 | 現状 | MPI 化 | リスク |
|---|---|---|---|
| エントリ | `HF_main.cu` main()、`MultiGpuManager::initialize()` | `MPI_Init/Finalize` 追加、local_rank で GPU pin（実質ランチャ任せ）、init を rank-aware に | 中 |
| GPU管理 | `MultiGpuManager` が N device 巡回 | MPI モード = 1 device/1 comm に縮退。`is_distributed()` の意味を「world_size>1」に拡張 | 中 |
| NCCL | `ncclCommInitAll` | `ncclCommInitRank` + Bcast(uniqueId) | 低〜中 |
| RI 分散 | `eri_ri_distributed.cu` が `for d in devices` で aux 軸分割し各 device に DGEMM + AllReduce | 「device ループ」→「自 rank の slab のみ計算 + world AllReduce」。`aux_partition(global,N,d)` の `d` を world_rank に | **高**（分散 RI の中核、要慎重移行） |
| stage-1 | `dlpno_mp2.cu` / `dlpno_pair_data.cu` の per-device pair 分配 | pair を world rank で分配 → 各 rank 自 GPU で solve → 結果 gather | 高 |
| stage-2 | IP/EA operator が device 0 基準で逐次 | **rank0=IP / rank1=EA 並列**、bar-H を NCCL send/recv で rank0 へ集約 | 高（G2） |
| 非分散経路 | RHF/post-HF の single-GPU 計算 | 各 rank が冗長に同じ計算 or rank0 のみ実行+Bcast。要方針決定（§5） | 中 |

---

## 4. 移行ステップ（最小から積み上げ、各段で検証）

各ステップは「single-node multiprocess で開発・検証 → multi-node はコード変更ゼロ」を満たす形で進める。
ビルド/実行はリモート。検証は corr / IP0 9.6861 / EA0 3.0983 の bit-level 一致を基準にする。

**Step 0 — ビルド基盤（低リスク、まず通す）**
- CMake に `find_package(MPI)` + `GANSU_MPI` define + `MPI::MPI_CXX` link を追加。
- `GANSU_MPI` 未定義時は **完全に現状コード**（`#ifdef GANSU_MPI` で全 MPI コードを gate）。
- 成果物: `GANSU_MPI` ビルドが `-np 1` で現状と byte-identical に走る（MPI_Init/Finalize だけ追加、他は no-op）。

**Step 1 — MPI scaffolding + NCCL-via-MPI（基盤）**
- `MpiEnv` 薄ラッパ（`include/mpi_env.hpp`）: `MPI_Init/Finalize`、`world_rank()`/`world_size()`/`local_rank()`。
- `MultiGpuManager::initialize()` を MPI 分岐: world_size>1 なら `ncclCommInitRank`+Bcast、1 comm 保持。
- ランチャ `script/gansu_mpi.sh`: `CUDA_VISIBLE_DEVICES=$LOCAL_RANK` 設定。
- 成果物: `mpirun -np 2` で 2 rank 起動・NCCL AllReduce 疎通テスト（小配列の sum が world で一致）。

**Step 2 — 分散 RI を MPI 化（最初の実計算）**
- `eri_ri_distributed.cu` の device ループを world-rank 分割に。`aux_partition(global, world_size, world_rank)`。
- 各 rank が自 slab の 3c2e + DGEMM → `nccl::all_reduce`（device_id=0）で world 集約。
- 成果物: RHF energy が `-np N` と single-process で bit 一致（H2O cc-pVDZ -76.024188... 基準）。

**Step 3 — stage-1 DLPNO-CCSD の MPI 化**
- pair を world rank で分配、各 rank 自 GPU で per-pair solve、結果を gather（host staging or CUDA-aware MPI）。
- 成果物: DLPNO-CCSD corr が `-np N` で single-process と一致（naphthalene corr -1.2624234101）。

**Step 4 — stage-2 STEOM 並列（G2、本丸）**
- rank0=IP solve / rank1=EA solve を同時起動（各自 GPU、operator device-0 前提が rank 内で成立）。
- EA の bar-H (Wvovv+Wvvvo ≈ 23GB) を rank1→rank0 へ NCCL send/recv。STEOM build_W_eff_and_G を rank0 で。
- 成果物: STEOM 5 root が逐次版と一致（D2h jitter 帯内）、wall が IP+EA 逐次 → max(IP,EA) に短縮。

**Step 5 — multi-node 検証（G3）**
- hostfile を渡すだけ。コード変更ゼロを確認。ノード間は NCCL（NVLink/IB）+ MPI(host)。

---

## 5. 確定事項（2026-06-11 user 決定）

1. **非分散計算の扱い** = **(a) 全 rank 冗長計算**。各 rank が同じ計算を独立実行。実装単純・正しさ優先。
   後で hot path のみ rank0+Bcast 化しうる。
2. **cross-rank 転送** = **NCCL send/recv**（既存 `nccl_comm.hpp` ラッパ流用、ノード内 NVLink で高速、
   コード一貫）。CUDA-aware MPI は不採用。
3. **着手** = **Step 0 から実装**（CMake + `#ifdef GANSU_MPI` gate、`-np 1` byte-identical）。
4. （継続論点・実装中に確定）`--num_gpus` の MPI モード意味付け、single-process NCCL 経路の長期保守。
   暫定 = 両モード保守（既存ユーザ保護）、`-np` が MPI モードのデバイス数一次ソース。

---

## 6. 実装ログ

### Step 0 — ビルド基盤 + scaffolding ✅ VALIDATED（2026-06-11、H200×2）
検証結果: `-np 1` banner 無し・energy -76.02349076684369、`-np 2` で rank0→GPU0/rank1→GPU1 pin、
各 rank 冗長計算で energy 一致（末尾 ~1e-13 は別GPU reduction 順序、想定内）。
- `include/mpi_env.hpp`（新規）: RAII `MpiEnv`。ctor で `MPI_Init` + world/local rank 算出
  （`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` で node-local rank）、dtor で `MPI_Finalize`。
  `GANSU_MPI` 未定義時は single-rank stub（world_rank=0/world_size=1/is_mpi=false）。
- `src/HF_main.cu`: `#include "mpi_env.hpp"` + `<cstdlib>`、main() 冒頭で `MpiEnv mpi(argc, argv);`
  （全 return パスで finalize）。`is_mpi()` 時のみ stderr に rank banner（`-np 1`/非MPI は無出力 = byte-identical）。
- `CMakeLists.txt`: `option(ENABLE_MPI ... OFF)` + gansu target に `find_package(MPI REQUIRED)` +
  `GANSU_MPI` define + `MPI::MPI_CXX` link（`ENABLE_MPI=ON` 時のみ）。
- `script/gansu_mpi.sh`（新規）: ランチャ。node-local rank（OMPI/MVAPICH2/Slurm/MPICH 各 env 対応）→
  `export CUDA_VISIBLE_DEVICES=$LOCAL_RANK` → `exec "$@"`。各 rank が自GPUを device 0 と認識。

**リモート検証手順（user）**:
```bash
# ビルド
mkdir build && cd build
cmake .. -DENABLE_MULTI_GPU=ON -DENABLE_MPI=ON && make
# (1) -np 1 で現状と同一に走ること（banner 無し、energy 一致）
./gansu -x ../xyz/H2O.xyz -g cc-pvdz -m RHF
# (2) -np 2 で 2 rank が別 GPU に pin されて起動すること（banner 2 行、energy は各 rank 同一 = 冗長計算）
mpirun -np 2 --bind-to none ../script/gansu_mpi.sh ./gansu -x ../xyz/H2O.xyz -g cc-pvdz -m RHF
#   期待 stderr:
#   [MPI] rank 0/2 (local 0/2, CUDA_VISIBLE_DEVICES=0)
#   [MPI] rank 1/2 (local 1/2, CUDA_VISIBLE_DEVICES=1)
# (3) 非 MPI ビルド（-DENABLE_MPI=OFF）が従来通りビルド・実行できること（回帰なし）
```
**Step 0 の不変条件**: `ENABLE_MPI=OFF` ビルドは従来と byte-identical。`ENABLE_MPI=ON` でも `-np 1` は
banner を出さず計算結果も同一。この段では NCCL は従来の `ncclCommInitAll` のまま（置換は Step 1）。

### Step 1 — MPI scaffolding + NCCL-via-MPI ✅ VALIDATED（2026-06-11、H200×4）
検証: `mpirun -np 4 ... --mpi_selftest` で 4 rank→GPU0-3 pin、
`[MPI] NCCL world AllReduce self-test: sum=4 (expected 4) OK`（cross-rank NCCL 疎通確認）、
RHF energy -76.0234907668 ×4 一致（~1e-13 reduction 差）。
- `include/multi_gpu_manager.hpp`: `world_rank_`/`world_size_` メンバ + `world_rank()`/`world_size()`/`is_mpi()`
  accessor 追加。`is_distributed()`（=num_devices>1）は intra-process 専用と明記、cross-rank は `is_mpi()`。
- `src/multi_gpu_manager.cu` `initialize()`:
  - `MPI_Initialized` → `MPI_Comm_rank/size` で world rank/size 取得。
  - `is_mpi()` 時 `num_devices_=1`（各 rank 自 GPU = device 0）。handle/stream は従来ループで 1 個生成。
  - NCCL: `is_mpi()` 分岐 = rank0 `ncclGetUniqueId` → `MPI_Bcast` → 全 rank `ncclCommInitRank(world_size, id, rank)`、
    `nccl_comms_[0]` に保持。非 MPI は従来 `ncclCommInitAll`。GANSU_MPI かつ NCCL 無し時は警告。
  - **AllReduce self-test**: 各 rank が 1.0 寄与 → world sum == world_size を rank0 が print。
  - 完了 print を MPI 時は rank0 のみ「N MPI rank(s) x 1 GPU, NCCL world comm enabled」に。
- `src/HF_main.cu`: `--mpi_selftest` フラグ。stored-RHF 等 init を呼ばない経路でも `MultiGpuManager::initialize()`
  を強制し self-test を確定実行（Step 1 検証フック）。

**リモート検証（user）**:
```bash
cd build && cmake .. -DENABLE_MULTI_GPU=ON -DENABLE_MPI=ON && make
mpirun -np 4 --bind-to none ../script/gansu_mpi.sh ./gansu -x ../xyz/H2O.xyz -g cc-pvdz -m RHF --mpi_selftest \
  2>&1 | grep -E '\[MPI\]|\[MultiGPU\]|Total Energy'
#   期待:
#   [MPI] rank 0..3 banner ×4
#   [MultiGPU] Initialized: 4 MPI rank(s) x 1 GPU (NVIDIA H200 NVL), NCCL world comm enabled
#   [MPI] NCCL world AllReduce self-test: sum=4 (expected 4) OK   ← 最重要（cross-rank NCCL 疎通）
#   Total Energy ... ×4（冗長計算、一致）
# 回帰: -np 1（--mpi_selftest 無し）と ENABLE_MPI=OFF が従来通り。
```
**不変条件**: `is_mpi()` でない実行（`-np 1`/非MPI）は NCCL 経路が従来 `ncclCommInitAll` のまま = 既存
single-process multi-GPU（`--num_gpus`）挙動不変。MPI self-test/print は rank0 のみで stdout 汚染最小。

### Step 2 以降 — 未着手
- Step 2: 分散 RI を world-rank 分割（`aux_partition(global, world_size, world_rank)` + AllReduce）。
- Step 3: stage-1 DLPNO-CCSD の pair を world rank 分配。Step 4: STEOM rank0=IP/rank1=EA。Step 5: multi-node。

---

## 7. 注意
- ⚠ RI 6ファイル（Term A）の user 手動コミットが先（メモ記載）。MPI 着手はそれと独立に進められる。
- ⚠ ビルド/検証はリモート GPU マシン専用。git 操作は user が実施（コミット文面のみ提示）。
