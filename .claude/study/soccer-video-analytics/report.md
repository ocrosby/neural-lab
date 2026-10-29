# Soccer Video Analytics — Landscape Survey (late 2025 / early 2026)

**Scope:** reference-grade survey of companies, open-source tools, academic benchmarks, and
licensing paths relevant to a solo developer wanting to build a neural pipeline that ingests
broadcast soccer video and emits per-player + per-team statistics (identification, tracking,
events, xG/xT/PPDA). Not a general sports-analytics overview.

**Verification date:** 2026-10-29. Fast-moving corporate claims verified via WebSearch; items
marked `[unverified]` were not confirmable to a current primary source during this pass.

---

## Part 1 — Industry landscape

### Pro / elite tier — opaque, expensive, data is the product

| Vendor | Owner (2026) | Signal source | What they actually sell |
|---|---|---|---|
| **Opta / Stats Perform** | Stats Perform (independent, Vista Equity Partners-backed) | Human annotators + proprietary AI on broadcast | 7.2 PB of event data across 3,900+ competitions; still the dominant brand for "Opta stats"; **extended as Premier League / Football DataCo official media-data partner Dec 2024** |
| **Genius Sports** | Public (NYSE: GENI) | Second Spectrum optical + AI tracking tech | **Premier League official betting-data partner through 2028-29 season**; also Premier League/EFL official tracking-data provider via Second Spectrum; competes with Stats Perform on scope but Opta still owns the editorial/media narrative |
| **Second Spectrum** | Genius Sports (acq. 2021, $200M) | In-stadium multi-camera optical | Fully absorbed; brand effectively retired into Genius' "Performance Studio" line (updated April 2025). FIFA-certified EPTS. Powers Premier League / MLS / NBA tracking feeds. |
| **StatsBomb** | **Hudl (acq. Aug 2024)** | Human-annotated events + 360 freeze-frames | Richer event spec than Opta (shot freeze-frames, pressure, body-part); has famously generous open-data program (see Part 6). Now part of Hudl Pro Suite with Wyscout + Sportscode. |
| **TRACAB** | **Electronic Arts (acq. FY26 Q1 2025)** | Multi-camera optical in-stadium | World's most-deployed optical tracker: 300+ stadia, 4,500+ matches/yr. FIFA-certified SAOT (semi-automated offside). EA acquisition is notable — the data is now owned by a game publisher, not a sports-data vendor. |
| **SkillCorner** | Independent (raised $60M growth round `[unverified exact date]`) | **Broadcast single-camera** tracking | The one to study for broadcast-only. Covers 130+ competitions incl. Premier League and MLS from broadcast feed alone, with "off-screen player location modelling" to synthesize full 22-player tracking. Released 10-match open dataset from 2024/25 A-League (see Part 6). |
| **Signality** | **Spiideo (acq. Dec 2024)** | Optical tracking | Swedish; absorbed into Spiideo's cloud sports platform in Jan 2025. Primarily football; some AFL / baseball. |
| **InStat** | **Hudl (acq. 2022)** | Human annotation | Football InStat largely migrated into Wyscout; InStat brand now carries basketball + ice hockey only inside Hudl. |
| **Wyscout** | Hudl (acq. 2019) | Human annotation + video | **Formally rebranded "Hudl Wyscout" May 2025**; AI-driven player evaluation and predictive scouting added. Primary football platform in the Hudl stack. |

**Ownership flux is heavy and recent.** The 2024–2025 window alone consolidated StatsBomb,
Signality, and TRACAB into larger owners (Hudl, Spiideo, EA respectively). The "big three"
data vendors are effectively **Stats Perform (Opta)**, **Genius Sports (Second Spectrum)**,
and **Hudl (Wyscout + StatsBomb + Sportscode)**. Everyone else is either a tech supplier to
them or a niche play.

### Semi-pro / mid-market — tools, not pure data

- **Hudl (parent)** — Sportscode (video tagging), Assist (manual annotation service), Focus
  (auto-camera for team sports). Hudl's own stack is tagging + workflow; the "AI data" story
  comes from the acquired Wyscout / StatsBomb / InStat pieces.
- **Metrica Sports** — tracking + event analysis platform. Historically important for its
  open sample dataset; corporate activity on their public GitHub has slowed (sample-data
  repo last commit April 2021, though a `more-than-a-hack-2027` repo was updated Oct 2026).
- **Catapult** — wearables (Vector S7/T7/X7); FIFA ETPS certified; GPS at 10Hz on outdoor
  units. Used by Real Madrid, Chelsea, etc. Pure sensor play — no computer vision.
- **PlayerMaker** — 6-axis boot-mounted sensors sampling at 7,000 Hz. FIFA-certified for
  match use (Sept 2023). Competes with Catapult on physical metrics but adds technical
  touches / ball-interaction data.

### Amateur / consumer-adjacent

- **Veo (Veo Technologies)** — **AI auto-tracking camera, verified 2026 pricing**:
  - Veo Cam 3: $1,299 hardware (5G variant $1,599)
  - Subscription tiers: Starter $42/mo → Enterprise $178/mo
  - Veo Go (phone-mount): $50 hardware + $29/mo
  - Youth/semi-pro market; subscription mandatory for cloud storage + AI processing.
- **Trace** — competing US-focused player: PlayerFocus Basic $180/yr, Pro $300/yr; annual
  plans bundle hardware. Positioned similarly to Veo but with stronger youth highlights focus.
- **Reeplayer, BallerCam** — adjacent smaller players.

### Broadcast-only vs multi-camera — the decisive distinction

| Vendor | Primary signal path |
|---|---|
| TRACAB, Second Spectrum, Signality | **In-stadium multi-camera rig** — needs physical install, controlled calibration |
| SkillCorner | **Broadcast single-feed** — can run on any broadcast video, no stadium access |
| Opta, StatsBomb | Hybrid — human annotators + algorithmic assists on broadcast |
| Veo, Trace | Dedicated single-camera auto-pan — amateur/youth, controlled vantage |

**For a solo developer, this distinction is everything.** You can only build what you can
feed. You will not have in-stadium multi-camera data; your pipeline must be broadcast-only.
That puts SkillCorner (not Second Spectrum, not TRACAB) as the right technical reference.

---

## Part 2 — Subtask breakdown — what Opta/SkillCorner are actually doing under the hood

1. **Camera calibration / field homography.** Map broadcast pixels to pitch coordinates
   (meters). Standard family: deep segment-prediction (DeepLabv3+) + differentiable
   calibration objective. **TVCalib (2022)** is the current baseline; **PnLCalib (2026)** and
   **TacticalCalib (WACV 2026)** are newer. Broadcast-only works; moving cameras make this
   the hardest step after action spotting. Must be re-estimated per frame or per cut.
2. **Player detection.** Standard object detection: YOLO v11/v12 (2024–2025), **RF-DETR
   (Roboflow, March 2025)** which first broke 60 mAP on COCO under 25 FPS and ships under
   Apache 2.0. Broadcast-only, easy — this is the solved part. Fine-tune on SoccerNet.
3. **Multi-object tracking with occlusion.** ByteTrack remains the common backbone;
   **BoT-SORT** adds appearance re-ID; **StrongSORT** improves association. **TrackTrack
   (CVPR 2025)** is now the Ultralytics default since v8.4.76 (June 2026). Soccer-specific
   twist: 22 players + ref(s) + ball + similar team uniforms = association is the hard part.
   SportsMOT and SoccerTrack specialize this.
4. **Jersey number OCR + team classification.** Persistent identity across occlusions. The
   **Koshkina et al. (CVPR 2024) framework** at `github.com/mkoshkina/jersey-number-pipeline`
   is the current open reference — pose-estimator-cropped torso region → fine-tuned PARSeq
   (scene-text recognition). Reports 87.4% soccer tracklet accuracy. Pure OCR on raw crops
   gets only ~36%. Team classification is typically k-means on jersey color crops.
5. **Ball detection + tracking.** Harder than players — 10–15 px at best, often occluded,
   motion-blurred, sometimes off-screen. Standard approach: slicing-aided detection (SAHI)
   + specialized small-object detector + separate tracker with anomaly handling (ball
   disappears for frames). Broadcast-only — multi-camera trivializes but is not required.
6. **Action spotting (temporal).** When did something happen? SoccerNet-BAS (Ball Action
   Spotting) 2024 expanded from 2 → 12 action classes (Pass, Drive, Header, High Pass, Out,
   Cross, Throw In, Shot, Ball Player Block, Successful Tackle, Free Kick, Goal) on 11,041
   annotated timestamps. Model family: video-feature extractor (I3D / TSM / VideoMAE) +
   temporal module (Transformer, dense anchors). Hardest subtask after calibration.
7. **Action classification.** What *kind* of event (pass short vs long, shot header vs
   footed)? Usually fused with action spotting as a joint head; otherwise a secondary
   classifier on windowed clip features.
8. **Pose estimation.** Finer events (fouls, body-part of shot). **RTMPose** has displaced
   OpenPose in 2025 for speed. **MMPose** has not released since July 2024 — slowing but not
   dead. **OpenPifPaf** is still the bottom-up go-to for crowded occluded scenes. Needed for
   top-tier events; a basic pipeline can skip it.
9. **Event aggregation → statistics.** Pure engineering — once events are emitted with
   `(player, action, pitch_xy, time)`, deriving per-match passes/tackles/shots/distance is
   SQL-level. No ML here.
10. **Advanced metrics (xG, xT, PPDA).** Classical ML on event streams. **xG** = logistic /
    GBM over shot features (distance, angle, body part, defender count, assist type) — Opta
    uses up to 20 context factors per shot. **xT** = value-surface model (Karun Singh, 2018)
    assigning value to pitch zones; progression value. **PPDA** = `opponent_passes_in_their_
    60% / own_defensive_actions_in_that_zone` — arithmetic, not ML. **Formation detection** =
    cluster average-positions-per-player over possession windows.

**Broadcast-only sufficiency:** steps 1–7 all work from a single broadcast feed; multi-camera
makes them better, not categorically different. Steps 8–10 are downstream of the event
stream and model-agnostic to signal source.

---

## Part 3 — Open-source stack a solo developer can actually use

Status key: **Active** = commits within 6 months; **Semi-active** = 6–18 months; **Stale** =
no commits in 18+ months.

### Benchmarks + datasets
- **SoccerNet** — `github.com/SoccerNet/sn-*` — **Active**. Dev kit MIT-licensed; raw video
  requires NDA (see Part 6). Sub-benchmarks: Action Spotting, Ball Action Spotting (BAS),
  Game State Reconstruction (GSR), Depth, Camera Calibration, Jersey Number, Re-ID, Tracking,
  Replay Grounding. **The single most important free resource in this space.**
- **SoccerNet-GSR** (2024) — 200×30s sequences, 9.37M line points for calibration, 2.36M
  athlete positions with role/team/jersey. First open sports GSR dataset.
- **SoccerTrack-v2** — `github.com/AtomScott/SoccerTrack-v2` — **Active**. Full-pitch
  panoramic + multi-view. MIT-licensed. Also on HuggingFace at `atomscott/soccertrack-v2`.
- **SportsMOT** — 240 sequences, 150K+ frames, 1.6M+ boxes across basketball/volleyball/
  football. The go-to MOT benchmark for team sports.

### Detection
- **YOLOv11** — Ultralytics, Sept 2024. **Active**. Fewer params (-22%), higher accuracy
  than v8; still the production recommendation.
- **YOLOv12** — Feb 2025. Attention-centric; NeurIPS 2025 paper. Academic project — the
  Ultralytics team themselves recommend v11 or v26 for production over v12 (training
  instability, higher memory, slower on CPU).
- **RF-DETR** — Roboflow, March 2025. **Active**. Apache 2.0. 6 sizes Nano → 2XL. First
  real-time (25+ FPS on T4) to break 60 mAP on COCO. Instance segmentation + keypoint
  variants shipped. Best current default for sports.

### Tracking (multi-object)
- **BoxMOT** — `github.com/mikel-brostrom/boxmot` — **Active** (updated Oct 2026, 8.3k★).
  Pluggable Python/C++ tracker library. The "integration layer" — single API over BoT-SORT,
  ByteTrack, StrongSORT, OC-SORT, etc.
- **Roboflow `trackers`** — **Active** (updated Oct 2026). Clean-room implementations of
  SORT / ByteTrack / OC-SORT / BoT-SORT / C-BIoU / McByte, benchmarked on MOT17 **and
  SoccerNet + SportsMOT + DanceTrack specifically**.
- **ByteTrack** — `github.com/ifzhang/ByteTrack` — Reference impl, MIT. Original repo
  semi-active but widely vendored into every tracker library.
- **BoT-SORT** — `github.com/NirAharon/BoT-SORT` — MIT. Standard "ByteTrack + appearance" baseline.
- **norfair** — `github.com/tryolabs/norfair` — **Active** (v2.3.0 released April 2025).
  Lightweight; works with any detector. Good for prototyping.

### Pose estimation
- **MMPose** — OpenMMLab. **Semi-active**: no release since July 2024, limited new activity.
  Still used because the model zoo is huge and the pretrained weights are canonical.
- **OpenPifPaf** — Still the bottom-up option for occluded crowded scenes; low compute budget.
  **Semi-active** `[unverified exact last-release date]`.
- **AlphaPose** — Legacy; largely superseded by MMPose/RTMPose for new work.
- **RTMPose** (inside MMPose) — Current speed-optimized choice; preferred over OpenPose.

### Soccer-specific pipelines + examples
- **SoccerNet dev kit** — the canonical starting point. Baseline implementations for every
  SoccerNet task.
- **Roboflow blog tutorials** — reference pipelines combining YOLO/RF-DETR + ByteTrack /
  BoT-SORT + `supervision` library for soccer specifically. Updated through 2025–2026.
- **mkoshkina/jersey-number-pipeline** — CVPR 2024 framework for jersey OCR. **Active**.
- **narita-takumi/SoccerTrack (v1)** has been superseded by **AtomScott/SoccerTrack-v2**;
  use v2.
- **MMAction2** — OpenMMLab. **Semi-active**: v1.2.0 was the last tagged release (Oct 2023
  per their changelog, but issue activity continues through 2026). Model zoo for TSN / TSM /
  I3D / VideoMAE used for action spotting features. Treat as "mature, use carefully, do not
  expect new models to land".
- **mmtracking** — **Stale**. Last commit April 2023; community has openly asked whether
  it's dead. OpenMMLab shifted focus to LLMs after Prof. Tang Xiaoou's passing in Dec 2023.
  Use BoxMOT or Roboflow trackers instead.
- **StatsBombR / statsbombpy** — R and Python clients for StatsBomb open event data (not
  video; see Part 6). **Active**.

---

## Part 4 — Academic benchmarks and challenges

- **SoccerNet 2022** — Action spotting, replay grounding, player re-ID, calibration, tracking.
- **SoccerNet 2023** — Same + ball action spotting introduced.
- **SoccerNet 2024** — BAS expanded to 12 classes; **Depth** benchmark introduced
  (SoccerNet-Depth, synthetic); **GSR** benchmark introduced (first open sports GSR data).
- **SoccerNet 2025** — 5th edition, 4 tracks: Team Ball Action Spotting, Monocular Depth
  Estimation, Multi-View Foul Recognition, Game State Reconstruction.
- **SoccerNet 2026** — 6th edition, 5 tracks: **Ball Action Anticipation** (future-window
  prediction, new), **Player-Centric Ball Action Spotting** (localize + classify + assign to
  specific player via team + jersey), **Novel View Synthesis**, **Spiideo SoccerNet Synloc**
  (athlete localization in pitch coords from single static-camera image), **Visual Question
  Answering** (text + image + video). 427 teams, 1,129 submissions.
- **SportsMOT** — MOT across football/basketball/volleyball; 240 seq / 150K frames / 1.6M
  boxes / 3,401 tracks. The team-sports MOT benchmark. Access: paper + HuggingFace dataset.
- **SoccerTrack-v2 / Challenge 2025** — Persistent player tracking as of 2025; GTATrack
  (Deep-EIoU + Global Tracklet Association) won with HOTA 0.60.
- **SoccerDB** `[unverified current state]` — older event-detection dataset; largely superseded by SoccerNet.

**Access:** SoccerNet labels and dev kit are MIT; raw broadcast video requires NDA sign for
research use (see Part 6). SoccerTrack-v2 is MIT including the HuggingFace mirror.
SportsMOT public via arxiv paper + HuggingFace.

---

## Part 5 — Realistic project phasing for a solo developer

### MVP (3–6 months solo, broadcast video in → basic stats out)

**Achievable, by sub-task:**

- Camera calibration — **yes**, use TVCalib pretrained + fine-tune on SoccerNet-GSR
- Player detection — **yes**, RF-DETR or YOLOv11 fine-tuned on SoccerNet
- Player tracking — **yes**, BoxMOT (BoT-SORT) on top of detector; expect ID switches at
  occlusions
- Team classification — **yes**, k-means on jersey HSV crops, trivial
- Jersey number — **partially**, use Koshkina pipeline; expect 60–85% tracklet accuracy
- Ball detection + tracking — **painful but doable**, specialized small-object detector
  with SAHI; expect many missed frames
- Action spotting — **partial**, SoccerNet baselines will give you Pass/Drive/Shot/Goal
  reasonably; the finer 12-class split from BAS 2024 is harder
- xG / xT / PPDA on **your own emitted events** — **yes**, closed-form math over events

**Scope the MVP as:** broadcast feed in → per-player tracking with team labels and
best-effort jersey IDs → coarse event stream (pass/shot/goal spotted) → basic xG / possession
% / PPDA at team level. Accept 60–80% of commercial accuracy. Target one match end-to-end.

### Research-grade (12–24 months, more work)

- Multi-view / novel view synthesis experiments (SoccerNet 2026 tracks)
- Fine-grained 12-class ball action spotting at competitive HOTA / mAP
- Game State Reconstruction end-to-end (broadcast → minimap with persistent IDs — see "From
  Broadcast to Minimap", CVPR 2025 CVsports workshop)
- Foul detection / multi-view foul recognition
- Jersey OCR that competes with the Koshkina baseline
- Depth estimation + 3D reconstruction

### Commercial-grade — gated on things you don't have

Three gates block commercial-grade from a solo developer, regardless of model quality:

1. **Proprietary multi-camera data.** Opta/Second Spectrum/TRACAB train on in-stadium
   multi-view rigs you cannot access. Broadcast-only is a strictly harder problem; expect
   to be 10–20 HOTA points behind at steady state.
2. **Annotator army.** Opta's edge is 7.2 PB of human-reviewed annotations. The models are
   not the moat — the labeled data is.
3. **Rights.** Reselling event streams derived from Premier League video is a legal
   non-starter unless you have a Football DataCo license, which currently sits exclusively
   with Stats Perform and Genius Sports through 2029.

**Competitive context:** **Veo already serves the amateur market at ~$1.3k hardware +
$42–178/month subscription.** A solo-built broadcast analytics pipeline that competes with
Veo on auto-highlights or Trace on youth-focused highlights has a credible wedge; competing
with Opta on elite data does not. Think "open-source Veo" or "self-hosted Hudl-lite for a
single team", not "replace Opta".

---

## Part 6 — Where to get training video legally

**The data gap is the hardest part of this problem.** Models are open; labeled training
video is not.

### Research-licensed
- **SoccerNet (preferred).** Labels fully open; **raw broadcast video requires signing an
  NDA** with the SoccerNet consortium. Covers 500+ full matches across top European leagues,
  2014–2017 mostly, with newer additions. Non-commercial research use. The GitHub dev kit
  is MIT. This is the right legal path for a learning-oriented solo project.
- **SoccerTrack-v2 (MIT, HuggingFace-hosted).** Full-pitch panoramic + multi-view clips.
  Smaller than SoccerNet but no NDA. Open commercial-ish terms (MIT on the code; verify
  dataset-specific terms per clip).
- **SportsMOT.** MOT benchmark; football among 3 sports. Academic use, accessible via paper
  + HuggingFace.

### Event data (NOT video)
- **StatsBomb Open Data** at `github.com/hudl/open-data` — **3,961 matches across 80
  competition-seasons / 24 competitions** incl. every 2018 and 2022 World Cup match, FA WSL,
  2023 Women's World Cup, selected Champions League / La Liga. 3,400+ events per match with
  xG, freeze frames, StatsBomb 360 data. **This is event data, not video** — invaluable for
  training downstream statistical models (xG, xT, pass completion models) but useless as
  video training data.
- **Metrica Sports sample-data** — 3 matches of anonymized tracking + event data (CSV / EPTS
  / JSON). Last commit April 2021 but stable and still useful as a tracking-data sanity check.

### Public broadcast via YouTube
- Highlights and some full-match uploads exist, but legal status is ambiguous → DMCA risk.
  **Rule of thumb: fine for throwaway experimentation on your laptop; not fine for
  publishing a dataset, blog post, or paper that includes derived frames.** Fair-use is
  weakest when the model's output substitutes for the original (which analytics does).
- Some national leagues upload their own highlights to YouTube under permissive terms —
  check the specific channel. League-run channels (MLS, Bundesliga official) are safer than
  third-party reuploads.

### Open leagues / CC-licensed footage
- Essentially none at commercial-quality broadcast. The closest permissive sources are:
  - A-League 2024/25: SkillCorner released 10 matches of **derived tracking data**, not
    video, open-source — useful for pipeline evaluation but not training the vision model.
  - Occasional NCAA / amateur footage CC-licensed on YouTube, but not broadcast quality.
- **Realistic path:** record a youth or amateur match yourself with a GoPro or phone + a
  Veo-style pole mount. You own the footage and the moral rights clear. Not broadcast
  framing but viable for a prototype.

### Recommendation for the user
Start with **SoccerNet (NDA signed) + StatsBomb Open Data (events) + SoccerTrack-v2 (MIT)**.
This triad gives you: broadcast video for the vision pipeline, rich event data for the
downstream statistical models, and a secondary MIT-licensed source for sanity checks and
anything you want to publish openly.

---

## Appendix — Terminology quick reference

- **MOT / HOTA / IDF1** — multi-object tracking metrics; HOTA is the current standard.
- **GSR** — Game State Reconstruction; broadcast → full-match minimap with persistent IDs.
- **BAS** — Ball Action Spotting (SoccerNet task).
- **EPTS** — Electronic Performance and Tracking Systems; FIFA certification standard.
- **SAOT** — Semi-Automated Offside Technology (FIFA-certified; TRACAB and Second Spectrum
  both certified).
- **xG / xT / PPDA** — see Part 2, item 10.
- **Freeze frame (StatsBomb)** — 360° positional snapshot at shot moment, used in advanced
  xG and shot models.
- **Football DataCo (FDC)** — Premier League / EFL joint venture that licenses official
  data rights; current partners through 2029 are Stats Perform (media/insights) and Genius
  Sports (betting + tracking).
