# Soccer video analytics — landscape and project guide

Reference for building a neural pipeline that ingests broadcast soccer video and emits per-player + per-team statistics (identification, tracking, events, xG/xT/PPDA). Written for a solo developer deciding whether and how to approach this as a project.

Last reviewed: 2026-10-29. Corporate ownership and open-source project status move fast in this space; re-verify before committing to any specific tool. Sources at the end.

Companion docs: `docs/tools-overview.md` (general NN tooling), `docs/huggingface.md` (HF ecosystem). This doc stands on top of both — it assumes you've read them.

## The decisive distinction: broadcast vs multi-camera

Before anything else, understand this: commercial soccer analytics splits cleanly into two signal paths.

| Signal path | Vendors | What a solo dev can do with it |
|---|---|---|
| **In-stadium multi-camera rig** | TRACAB, Second Spectrum, Signality | Nothing — you don't have stadium access |
| **Broadcast single feed** | SkillCorner (the one to study), Opta (hybrid) | Everything your project will actually ship |
| **Dedicated auto-pan camera** | Veo, Trace | Semi-pro / amateur market; competitive floor you'd ship against |

**For a solo developer, you build broadcast-only.** The entire rest of this doc proceeds from that constraint. SkillCorner (Paris, independent) is the right technical reference because they publish on broadcast-only; Second Spectrum and TRACAB are not, no matter how impressive their results.

## Industry landscape

### Pro / elite tier — opaque, expensive, data is the product

| Vendor | Owner (2026) | What they sell |
|---|---|---|
| **Opta / Stats Perform** | Stats Perform (Vista Equity Partners) | 7.2 PB of human-annotated event data across 3,900+ competitions. Extended as Premier League / Football DataCo official **media-data** partner Dec 2024. |
| **Genius Sports** | Public (NYSE: GENI) | Owns Second Spectrum optical+AI tracking tech. Premier League official **betting + tracking** partner through 2028–29. |
| **Second Spectrum** | Genius Sports (acq. 2021, $200M) | In-stadium multi-camera optical. Brand folded into Genius' "Performance Studio" line (April 2025). FIFA-certified EPTS. Powers Premier League / MLS / NBA tracking feeds. |
| **StatsBomb** | **Hudl (acq. Aug 2024)** | Human-annotated events + 360° freeze-frames. Richer event spec than Opta (pressure, body-part, shot freeze-frames). Famously open-data-friendly — see "Where to get training video legally" below. |
| **TRACAB** | **Electronic Arts (acq. FY26 Q1 2025)** | Most-deployed optical tracker: 300+ stadia, 4,500+ matches/year. FIFA-certified SAOT (semi-automated offside). Now owned by a video-game publisher, not a sports-data vendor. |
| **SkillCorner** | Independent | **Broadcast single-camera** tracking. Covers 130+ competitions including Premier League and MLS from broadcast feed alone, with off-screen player location modelling to synthesize full 22-player tracking. |
| **Signality** | **Spiideo (acq. Dec 2024)** | Optical tracking. Swedish; absorbed into Spiideo's cloud platform Jan 2025. |
| **InStat** | **Hudl (acq. 2022)** | Football InStat merged into Wyscout. InStat brand now covers basketball + ice hockey only inside Hudl. |
| **Wyscout** | Hudl (acq. 2019) | Human annotation + video. **Rebranded "Hudl Wyscout" May 2025**; AI-driven player evaluation added. |

**The 2024–2025 window was unusually active.** StatsBomb → Hudl, Signality → Spiideo, TRACAB → EA all closed inside 18 months. The effective "big three" data vendors are now **Stats Perform (Opta)**, **Genius Sports (Second Spectrum)**, and **Hudl (Wyscout + StatsBomb + Sportscode)**. Everyone else is either a tech supplier to them or a niche play.

### Semi-pro / mid-market — tools, not pure data

- **Hudl (parent)** — Sportscode (video tagging), Assist (manual annotation service), Focus (auto-camera for team sports). Hudl's own stack is tagging + workflow; the AI data story comes from the acquired pieces.
- **Metrica Sports** — tracking + event analysis platform. Historically important for its open sample dataset; corporate activity slowed after 2021.
- **Catapult** — wearables (Vector S7/T7/X7); FIFA EPTS certified; GPS at 10 Hz. Used by Real Madrid, Chelsea. Pure sensor play — no computer vision.
- **PlayerMaker** — 6-axis boot-mounted sensors sampling at 7,000 Hz. FIFA-certified for match use (Sept 2023). Technical touches + ball-interaction data.

### Amateur / consumer-adjacent — your competitive floor

- **Veo (Veo Technologies)** — AI auto-tracking camera.
  - Veo Cam 3: $1,299 hardware (5G variant $1,599)
  - Subscription: Starter $42/mo → Enterprise $178/mo
  - Veo Go (phone-mount): $50 hardware + $29/mo
  - Youth / semi-pro market; subscription mandatory for cloud + AI processing.
- **Trace** — US-focused: PlayerFocus Basic $180/yr, Pro $300/yr. Positioned similarly to Veo with stronger youth-highlights focus.

**Position your project honestly:** Veo and Trace already serve the amateur market well. A solo-built broadcast analytics pipeline has a credible wedge as "open-source Veo" or "self-hosted Hudl-lite for a single team"; it does not have a credible wedge as "replace Opta."

## The actual subtask stack

What Opta or SkillCorner is doing under the hood, in pipeline order. Each is its own research problem.

### 1. Camera calibration / field homography

Map broadcast pixels to pitch coordinates in meters. Standard family: deep segment-prediction (DeepLabv3+) + differentiable calibration objective.

- **TVCalib** (2022) — current open baseline.
- **PnLCalib** (2026), **TacticalCalib** (WACV 2026) — newer approaches.

Must be re-estimated per frame or per broadcast cut. Hardest step after action spotting.

### 2. Player detection

Standard object detection. Mostly a solved tooling problem.

- **RF-DETR** (Roboflow, March 2025) — first real-time detector past 60 mAP on COCO under 25 FPS, Apache 2.0. **Best default for sports.**
- **YOLOv11** (Sept 2024) — Ultralytics production recommendation; fewer params than v8, higher accuracy.
- **YOLOv12** (Feb 2025) — attention-centric, NeurIPS 2025 paper. **Academic, not production** — Ultralytics themselves recommend v11 for production use.

Fine-tune on SoccerNet.

### 3. Multi-object tracking with occlusion

**The actual hard problem.** 22 players + ref(s) + ball + similar team uniforms = association is hostile.

- **ByteTrack** — standard backbone, MIT-licensed.
- **BoT-SORT** — ByteTrack + appearance re-ID; strong baseline.
- **StrongSORT** — improves association via richer motion/appearance fusion.
- **TrackTrack** (CVPR 2025) — Ultralytics default since v8.4.76 (June 2026).
- **BoxMOT** (8.3k★, actively maintained Oct 2026) — pluggable library with one API over all the above.

### 4. Jersey number OCR + team classification

How you get **persistent identity** across occlusions.

- **Koshkina et al. (CVPR 2024)** at `github.com/mkoshkina/jersey-number-pipeline` — pose-cropped torso → fine-tuned PARSeq scene-text recognition. Reports **87.4% tracklet accuracy on soccer**. Pure OCR on raw crops gets only ~36%.
- **Team classification** — k-means on jersey color crops. Simple and effective.

### 5. Ball detection + tracking

Harder than players — ball is 10–15 px at best, often occluded, motion-blurred, sometimes off-screen.

Standard approach: slicing-aided detection (**SAHI**) + specialized small-object detector + separate tracker with anomaly handling (ball disappears for frames at a time).

### 6. Action spotting (temporal)

*When* did something happen? **SoccerNet-BAS (Ball Action Spotting)** 2024 expanded from 2 → 12 action classes (Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Ball Player Block, Successful Tackle, Free Kick, Goal) on 11,041 annotated timestamps.

Model family: video-feature extractor (I3D / TSM / VideoMAE) + temporal module (Transformer, dense anchors). Hardest subtask after calibration.

### 7. Action classification

*What kind* of event? Pass short vs long, shot header vs footed. Usually fused with action spotting as a joint head; otherwise a secondary classifier on windowed clip features.

### 8. Pose estimation

Needed for fine-grained events (fouls, body-part of shot). A basic pipeline can skip this.

- **RTMPose** (inside MMPose) — displaced OpenPose in 2025 for speed.
- **MMPose** — no release since July 2024; slowing but model zoo is still canonical.
- **OpenPifPaf** — bottom-up option, best in crowded occluded scenes with limited compute.

### 9. Event aggregation → statistics

Pure engineering, no ML. Once events are emitted with `(player, action, pitch_xy, time)`, deriving per-match passes / tackles / shots / distance is SQL-level.

### 10. Advanced metrics (xG, xT, PPDA)

**Classical ML on event streams**, not video models.

- **xG** — logistic regression or GBM over shot features (distance, angle, body part, defender count, assist type). Opta uses up to 20 context factors per shot.
- **xT (expected threat)** — Karun Singh (2018) value-surface model assigning value to pitch zones; progression value.
- **PPDA** — `opponent_passes_in_their_60% / own_defensive_actions_in_that_zone`. Arithmetic, not ML.
- **Formation detection** — cluster average-positions-per-player over possession windows.

Train these on **StatsBomb Open Data** — see the data section below.

**Broadcast-only sufficiency:** steps 1–7 all work from a single broadcast feed; multi-camera makes them better, not categorically different. Steps 8–10 are downstream of the event stream and model-agnostic to signal source.

## Open-source stack you can actually use

Status key: **Active** = commits within 6 months; **Semi-active** = 6–18 months; **Stale** = no commits in 18+ months.

### Benchmarks + datasets

- **SoccerNet** — the single most important free resource. Dev kit MIT; raw video requires NDA (see data section). Sub-benchmarks: Action Spotting, Ball Action Spotting (BAS), Game State Reconstruction (GSR), Depth, Camera Calibration, Jersey Number, Re-ID, Tracking, Replay Grounding. **Active.**
- **SoccerNet-GSR** (2024) — 200 × 30-second sequences, 9.37M line points for calibration, 2.36M athlete positions with role/team/jersey. First open sports GSR dataset.
- **SoccerTrack-v2** — `github.com/AtomScott/SoccerTrack-v2`. Full-pitch panoramic + multi-view, MIT. Also on HuggingFace at `atomscott/soccertrack-v2`. **Active.**
- **SportsMOT** — 240 sequences, 150K+ frames, 1.6M+ boxes across basketball / volleyball / football. The team-sports MOT benchmark.

### Detection

- **RF-DETR** (Roboflow) — Apache 2.0, 6 sizes Nano → 2XL. Best current default for sports. **Active.**
- **YOLOv11** (Ultralytics) — production recommendation. **Active.**

### Tracking

- **BoxMOT** — pluggable Python/C++ tracker library. One API over BoT-SORT, ByteTrack, StrongSORT, OC-SORT. **Active (Oct 2026, 8.3k★).**
- **Roboflow `trackers`** — clean-room implementations benchmarked on SoccerNet + SportsMOT + DanceTrack. **Active (Oct 2026).**
- **norfair** — lightweight prototyping tracker; works with any detector. **Active (v2.3.0 April 2025).**

### Pose

- **MMPose / RTMPose** — model zoo + speed-optimized default. **Semi-active.**
- **OpenPifPaf** — occluded-scene fallback. **Semi-active.**

### Soccer-specific pipelines

- **SoccerNet dev kit** — canonical starting point; baseline implementations for every SoccerNet task.
- **Roboflow blog tutorials** — reference pipelines combining RF-DETR + ByteTrack / BoT-SORT + the `supervision` library for soccer specifically, updated through 2025–2026.
- **mkoshkina/jersey-number-pipeline** — jersey OCR, CVPR 2024. **Active.**
- **MMAction2** — model zoo for TSN / TSM / I3D / VideoMAE used in action spotting. **Semi-active** — mature, use carefully, don't expect new models.
- **mmtracking** — **Stale** (last commit April 2023). OpenMMLab pivoted to LLMs after Prof. Tang Xiaoou's passing Dec 2023. Use BoxMOT or Roboflow trackers instead.

## Academic benchmarks

- **SoccerNet 2022** — Action spotting, replay grounding, player re-ID, calibration, tracking.
- **SoccerNet 2023** — + ball action spotting introduced.
- **SoccerNet 2024** — BAS expanded to 12 classes; Depth benchmark introduced (synthetic); GSR benchmark introduced.
- **SoccerNet 2025** — 5th edition, 4 tracks: Team Ball Action Spotting, Monocular Depth, Multi-View Foul Recognition, Game State Reconstruction.
- **SoccerNet 2026** — 6th edition, 5 tracks: Ball Action Anticipation (new), Player-Centric Ball Action Spotting (localize + classify + assign to specific player), Novel View Synthesis, Spiideo SoccerNet Synloc (athlete localization from single static-camera image), Visual Question Answering. 427 teams, 1,129 submissions.
- **SportsMOT** — MOT across football / basketball / volleyball; 240 seq / 150K frames / 1.6M boxes / 3,401 tracks.
- **SoccerTrack-v2 Challenge 2025** — Persistent player tracking. GTATrack (Deep-EIoU + Global Tracklet Association) won with HOTA 0.60.

## Dataset verdict — HuggingFace soccer-datasets page 3

The 28 datasets on the trending-sorted page 3 of HuggingFace's soccer-dataset search, assessed against the subtask stack above.

### Directly useful — pull these

| Dataset | Pipeline stage it serves |
|---|---|
| `SoccerNet/SN-Calibration` + `SoccerNet/SN-Calibration-2023` | Stage 1 — camera calibration. Essential. |
| `SoccerNet/SN-Tracking-2023` | Stages 2 + 3 — player tracking labels. Essential. |
| `SoccerNet/SN-ReID-2023` | Stage 3 — player re-ID across occlusions. Essential. |
| `bambezius/soccernet-v3-ball-tracking` | Stage 5 — ball-specific tracking. |
| `SoccerNet/SN-BAS-2023` | Stage 6 — ball action spotting (temporal). |
| `SoccerNet/SN-Labels` | Stage 7 — event classification labels. |
| `SoccerNet/FoulRet` + `OpenSportsLab/SoccerNet-FoulRet` | Stage 7 — foul detection subtask. |

### Operationally useful

- `SoccerNet/SN-Features` — pre-extracted frame features. Skip video decode when training downstream heads. Not training data per se.

### Mirrors / forks — pull the official version instead

`mrhwl123/SoccerNet2022`, `mrhwl123/SoccerNet2023`, `bambezius/soccernet-tracking`, `bambezius/soccernet-calibration-2023`, `rigan/soccernet-ball-action-spotting` — all community re-uploads of official SoccerNet assets. Use the `SoccerNet/` namespace.

### Marginal — specific use cases only

- `SoccerNet/SN-Depth` — monocular depth; not standard for stats pipelines.
- `SoccerNet/SN-VQA-2027` — visual QA; different task.
- `MonkeyBoy999/soccernet_closeups_998` — close-up frames; niche subset.
- `algo-dev99/soccertrack-v2-followcam-events` — follow-cam-specific; broadcast feeds don't look like this.
- `qleandataset/video-soccer-match` (28 rows), `Lya111/PDTP_Soccer_Train` (43 rows) — too small for anything.

### Not useful for a video pipeline

- The four `*/soccer-dataset` clones at 17.1M rows each (`eatpizzanot`, `svgwvgs`, `Sporoseseun`, `Oscar343`) — **tabular event data**, almost certainly a mirror of StatsBomb Open Data. Useful for training **xG and xT heads** at stage 10, not for training a video model. Pull the official StatsBomb repo instead.
- `dahara1/soccerscope-worldcup2026` — 928 rows tabular. Not video.
- `tsangb34/robocolosseum-so101-soccer-red_bowl-40episodes` — SO-101 robot arm. Not soccer.

### What's missing from the HF page that you'll need

- **SoccerNet-GSR (Game State Reconstruction)** — the 2024/2025 benchmark combining calibration + tracking + identification + action spotting end-to-end. On SoccerNet's own site.
- **Raw broadcast video** — SoccerNet distributes it under NDA at `soccer-net.org`, not through HuggingFace.
- **Jersey-number OCR data** — Koshkina's `jersey-number-pipeline` on GitHub ships with its own data.
- **Pose annotations on soccer** — nothing specific; use MMPose/OpenPifPaf pretrained on COCO as a starting point.

## Realistic project phasing

### MVP — 3–6 months solo, broadcast video in → basic stats out

| Stage | Tool | Expectation |
|---|---|---|
| Calibration | TVCalib pretrained + fine-tune on SoccerNet-GSR | Works |
| Detection | RF-DETR fine-tuned on SoccerNet | Works |
| Tracking | BoxMOT (BoT-SORT) | ID switches at occlusions expected |
| Team classification | k-means on jersey HSV crops | Trivial |
| Jersey number | Koshkina pipeline | 60–85% tracklet accuracy |
| Ball | RF-DETR + SAHI | Many missed frames expected |
| Action spotting | SoccerNet baselines | Pass / Drive / Shot / Goal reasonably; the finer 12-class split from BAS 2024 is harder |
| xG / xT / PPDA | Classical models on StatsBomb Open Data | Works |

**Scope the MVP as:** broadcast feed in → per-player tracking with team labels and best-effort jersey IDs → coarse event stream (pass/shot/goal spotted) → basic xG + possession % + PPDA at team level. Accept 60–80% of commercial accuracy. **Target one match end-to-end**, not every match.

### Research-grade — 12–24 months

- Multi-view / novel view synthesis (SoccerNet 2026 tracks)
- Fine-grained 12-class ball action spotting at competitive HOTA / mAP
- Game State Reconstruction end-to-end
- Foul detection / multi-view foul recognition
- Jersey OCR that competes with the Koshkina baseline
- Depth estimation / 3D reconstruction

### Commercial-grade — gated on things you don't have

Three gates block commercial-grade from a solo developer, regardless of model quality:

1. **Proprietary multi-camera data.** Opta / Second Spectrum / TRACAB train on in-stadium multi-view rigs. Broadcast-only is strictly harder; expect to be 10–20 HOTA points behind at steady state.
2. **Annotator army.** Opta's edge is 7.2 PB of human-reviewed annotations. The models are not the moat — the labeled data is.
3. **Rights.** Reselling event streams derived from Premier League video is a legal non-starter without a Football DataCo license, which currently sits exclusively with Stats Perform and Genius Sports through 2029.

## Where to get training video legally

The data gap is the hardest part. Models are open; labeled training video is not.

### Research-licensed

- **SoccerNet (preferred).** Labels open. **Raw broadcast video requires signing an NDA** with the SoccerNet consortium. Covers 500+ full matches across top European leagues, mostly 2014–2017 with newer additions. Non-commercial research use. Dev kit MIT. This is the right legal path for a learning-oriented solo project.
- **SoccerTrack-v2 (MIT, HuggingFace-hosted).** Full-pitch panoramic + multi-view clips. Smaller than SoccerNet but no NDA.
- **SportsMOT.** Academic use via paper + HuggingFace. Football among 3 sports.

### Event data (not video)

- **StatsBomb Open Data** at `github.com/hudl/open-data` — 3,961 matches across 80 competition-seasons / 24 competitions. Includes every 2018 and 2022 World Cup match, FA WSL, 2023 Women's World Cup, selected Champions League / La Liga. 3,400+ events per match with xG, freeze frames, StatsBomb 360 data. **Essential for training downstream statistical models (xG, xT, pass completion). Useless as video training data.**
- **Metrica Sports sample-data** — 3 matches of anonymized tracking + event data (CSV / EPTS / JSON). Last commit April 2021 but stable; good for tracking-data sanity checks.

### YouTube broadcast

Highlights and some full matches exist, but **legal status is ambiguous → DMCA risk**. Rule of thumb: fine for throwaway experimentation on your laptop; not fine for publishing a dataset, blog post, or paper that includes derived frames. Fair-use is weakest when the model's output substitutes for the original (which analytics does). League-run channels (MLS, Bundesliga official) are safer than third-party reuploads.

### CC-licensed footage

Essentially none at commercial-quality broadcast. The closest permissive sources:

- A-League 2024/25: SkillCorner released 10 matches of **derived tracking data**, not video. Useful for pipeline evaluation but not for training the vision model.
- Occasional NCAA / amateur footage CC-licensed on YouTube, but not broadcast framing.
- **Realistic alternative:** record a youth or amateur match yourself with a GoPro or phone on a Veo-style pole mount. You own the footage, moral rights clear. Not broadcast framing but viable for a prototype.

### Recommendation

Start with **SoccerNet (NDA signed) + StatsBomb Open Data (events) + SoccerTrack-v2 (MIT)**. This triad gives you:

- Broadcast video for the vision pipeline (SoccerNet)
- Rich event data for the downstream statistical models (StatsBomb)
- A secondary MIT-licensed source for sanity checks and anything you want to publish openly (SoccerTrack-v2)

## Relationship to the neural-lab ladder

This is a **parallel project**, not a ladder stage. It uses PyTorch (project 04+) and tracking / action-spotting models that build on everything up through transformers (07). Treat it as the applied project you'd take on *after* project 09 — before that, you don't yet have the modeling foundation to do it well.

Specifically, the subtask stack maps onto skills built in the ladder as follows:

| Subtask | Minimum ladder stage |
|---|---|
| Player detection, team classification | 05 (cnn-vision) — straightforward CNN work |
| Multi-object tracking | 06 (rnn-lm) + practical MOT library work; your learning covers the temporal modeling intuition |
| Action spotting, event classification | 07b (transformer-pytorch) — the attention mechanism matters for video-feature + temporal module designs |
| xG / xT models | 04 (pytorch-mlp) is sufficient — these are tabular |
| Full pipeline end-to-end | 09+ — once you've trained a GPT-shaped network, orchestrating a multi-model pipeline is familiar |

## Terminology quick reference

- **MOT / HOTA / IDF1** — multi-object tracking metrics; HOTA is the current standard.
- **GSR** — Game State Reconstruction; broadcast → full-match minimap with persistent IDs.
- **BAS** — Ball Action Spotting (SoccerNet task).
- **EPTS** — Electronic Performance and Tracking Systems; FIFA certification standard.
- **SAOT** — Semi-Automated Offside Technology; FIFA-certified (TRACAB and Second Spectrum both certified).
- **xG** — expected goals; probability a shot becomes a goal given shot context.
- **xT** — expected threat; value-surface model assigning value to pitch zones.
- **PPDA** — passes allowed per defensive action; pressing-intensity metric.
- **Freeze frame (StatsBomb)** — 360° positional snapshot at shot moment, used in advanced xG and shot models.
- **Football DataCo (FDC)** — Premier League / EFL joint venture that licenses official data rights; current partners through 2029 are Stats Perform (media / insights) and Genius Sports (betting + tracking).

## See also

- `docs/tools-overview.md` — general NN tooling across the neural-lab ladder.
- `docs/huggingface.md` — HF ecosystem orientation.
- `.claude/study/soccer-video-analytics/report.md` — the raw research report underlying this doc, with inline source URLs.

## Primary sources (verified during research)

- Stats Perform / Football DataCo partnership: `statsperform.com/insights/expanded-partnership-with-football-dataco/`
- Genius Sports / Football DataCo extension through 2029: `geniussports.com/newsroom/genius-sports-and-football-dataco-extend-exclusive-official-data-partnership-through-2029/`
- Hudl acquires StatsBomb (Aug 2024): `businesswire.com/news/home/20240812879329/en/`
- EA acquires TRACAB (Feb 2025): `sportsvideo.org/2025/02/04/electronic-arts-acquires-tracab-technologies/`
- Spiideo acquires Signality: `arcticstartup.com/spiideo-acquires-signality/`
- SkillCorner broadcast tracking: `skillcorner.com/sports/football`
- SoccerNet 2026 challenges: `arxiv.org/html/2607.07320`
- SoccerNet FAQ (license, NDA): `soccer-net.org/faq`
- StatsBomb Open Data repo: `github.com/hudl/open-data`
- RF-DETR announcement: `blog.roboflow.com/rf-detr/`
- YOLOv12 NeurIPS 2025 repo: `github.com/sunsmarterjie/yolov12`
- Veo Cam 3 pricing: `veo.com/en-us/veo-cam-3-starter-subscription-product-page`
- mmtracking "is this repo dead?" issue: `github.com/open-mmlab/mmtracking/issues/934`
- Koshkina jersey-number pipeline: `github.com/mkoshkina/jersey-number-pipeline`
- SoccerTrack-v2: `github.com/AtomScott/SoccerTrack-v2`
- TVCalib paper: `arxiv.org/pdf/2207.11709`
