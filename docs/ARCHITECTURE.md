# YT Did You Know - Architecture & Flow Diagrams

## High-Level System Architecture

```mermaid
graph TB
    subgraph "External Sources"
        YT[YouTube Data API]
        RD[Reddit API]
        GH[GitHub Trending]
        HN[Hacker News]
        HF[Hugging Face Daily Papers]
        HFH[HF Hub Models/Datasets]
        AX[ArXiv API]
        GT[Google Trends RSS]
        MD[Medium RSS Feeds]
    end

    subgraph "Trending Engine"
        TE[fetch_all_trending_signals]
        SR[Source Rotation Tracker]
        SC[Engagement Scoring]
    end

    subgraph "Topic Intelligence"
        ED[Embedding Deduplication<br/>sentence-transformers]
        ZS[Zero-Shot Classification<br/>facebook/bart-large-mnli]
        TC[Topic Classifier]
    end

    subgraph "Content Pipeline"
        GS[Gemini Script Gen]
        AV[Audio Generation]
        CB[Chunk Builder]
        VF[Visual Fetcher]
        VG[Video Generator]
    end

    subgraph "Distribution"
        YTU[YouTube Upload]
        IG[Instagram Reels]
        TT[TikTok/Threads]
        FB[Facebook]
        TG[Telegram]
    end

    YT --> TE
    RD --> TE
    GH --> TE
    HN --> TE
    HF --> TE
    HFH --> TE
    AX --> TE
    GT --> TE
    MD --> TE

    TE --> SR
    SR --> SC
    SC --> TC
    TC --> ED
    TC --> ZS
    ED --> GS
    ZS --> GS

    GS --> AV
    GS --> CB
    CB --> VF
    VF --> VG
    AV --> VG

    VG --> YTU
    VG --> IG
    VG --> TT
    VG --> FB
    VG --> TG
```

---

## Topic Selection & Classification Flow

```mermaid
flowchart TD
    Start([Start Pipeline]) --> Fetch[Fetch from All Sources]
    Fetch --> Score[Compute Base Engagement Scores]
    Score --> Rotate{Source Rotation<br/>Enabled?}
    
    Rotate -->|Yes| Boost[Apply Priority Boost<br/>1.5x for unused sources]
    Rotate -->|No| Sort[Sort by Base Score]
    Boost --> Sort
    
    Sort --> TopN[Take Top N Candidates]
    TopN --> LoadHist[Load Recent Topics<br/>from Tracker]
    LoadHist --> Embed[Generate Embeddings<br/>all-MiniLM-L6-v2]
    Embed --> Dedup{Similarity<br/>> 0.75?}
    
    Dedup -->|Yes| Reject[Reject Candidate<br/>Log Similar Topic]
    Dedup -->|No| Classify[Zero-Shot Classification<br/>facebook/bart-large-mnli]
    
    Reject --> NextCandidate{More<br/>Candidates?}
    Classify --> Confident{Confidence<br/>> 0.5?}
    
    Confident -->|Yes| Assign[Assign Predicted<br/>Category]
    Confident -->|No| Default[Default: AI & Tech Tools]
    
    Assign --> Filtered[Add to Filtered List]
    Default --> Filtered
    Filtered --> NextCandidate
    
    NextCandidate -->|Yes| Dedup
    NextCandidate -->|No| SortFinal[Sort by Boosted Score]
    SortFinal --> Output[Return Ranked Topics]
    Output --> End([End Topic Selection])
```

---

## Video Generation Pipeline

```mermaid
flowchart TD
    Script([Script Data]) --> Slot{Slot Type?}
    
    Slot -->|Shorts| ShortPath[Shorts Pipeline]
    Slot -->|Longform| LongPath[Longform Pipeline]
    
    subgraph ShortPath
        SP1[Generate Voiceover<br/>ElevenLabs/Edge-TTS]
        SP2[Stable-TS Word Timestamps]
        SP3[Build Visual Chunks<br/>Gemini subtitle_chunks]
        SP4[Fetch Visuals<br/>Pexels/HF/Imagen]
        SP5[Render Frames<br/>MoviePy + Custom Compositor]
        SP6[Add Kinetic Captions<br/>Day-style Variety]
        SP7[Add Retention Hooks<br/>Snap-zooms, SFX, Glitch]
        SP8[Encode MP4<br/>H.264/HEVC]
    end
    
    subgraph LongPath
        LP1[Chapter-Aware Chunking]
        LP2[Screenshot Capture<br/>Article/GitHub Evidence]
        LP3[Avatar Video Selection]
        LP4[Longform Compositor<br/>Talking Head + B-Roll]
        LP5[Progress Bar + CTA]
        LP6[Encode Long MP4]
    end
    
    ShortPath --> Upload
    LongPath --> Upload
    
    Upload[Upload & Distribute] --> YT[YouTube API]
    Upload --> IG[Instagram Graph API]
    Upload --> TT[TikTok/Threads API]
    Upload --> FB[Facebook Graph API]
    Upload --> TG[Telegram Bot]
    
    YT --> Notify[Telegram Notification]
    IG --> Notify
    TT --> Notify
    FB --> Notify
    TG --> Notify
    Notify --> Done([Video Live])
```

---

## Source Rotation State Machine

```mermaid
stateDiagram-v2
    [*] --> Fresh: Pipeline Start
    
    Fresh --> FetchMedium: Priority 1
    FetchMedium --> FetchGitHub: Priority 2
    FetchGitHub --> FetchHFHub: Priority 3
    FetchHFHub --> FetchArXiv: Priority 4
    FetchArXiv --> FetchYouTube: Priority 5
    FetchYouTube --> FetchHN: Priority 6
    FetchHN --> FetchHF: Priority 7
    FetchHF --> FetchGT: Priority 8
    FetchGT --> FetchYTOutliers: Priority 9
    FetchYTOutliers --> FetchYTPopular: Priority 10
    FetchYTPopular --> FetchReddit: Priority 11
    FetchReddit --> ApplyBoost
    
    ApplyBoost --> CheckHistory: Load Rotation State
    CheckHistory --> CalcBoost: For Each Source
    CalcBoost --> Boost15: If Not in Recent Window
    CalcBoost --> Boost10: If In Recent Window
    Boost15 --> SortArticles
    Boost10 --> SortArticles
    SortArticles --> SelectTop
    
    SelectTop --> RecordUsage: Save Source History
    RecordUsage --> TrimHistory: Keep Last 100
    TrimHistory --> ReturnResults
    ReturnResults --> [*]
    
    note right of CalcBoost
        Recent Window = 10 videos
        Sources per video ≈ 3
        History scan = 30 entries
    end note
```

---

## Embedding Deduplication Sequence

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant TC as TopicClassifier
    participant ST as SentenceTransformer
    participant TR as Tracker
    participant ZS as ZeroShotClassifier
    
    P->>TC: filter_and_classify_candidates(candidates, recent_topics)
    TC->>TR: get_recent_topic_embeddings(limit=30)
    TR-->>TC: List[Dict{title, description}]
    
    loop For each candidate
        TC->>ST: encode(candidate_title + description)
        ST-->>TC: embedding_vector (384-dim)
        
        loop For each recent_topic
            TC->>ST: encode(recent_title + description)
            ST-->>TC: recent_embedding
            TC->>TC: cosine_similarity(candidate, recent)
        end
        
        alt max_similarity > 0.75
            TC->>TC: reject candidate (duplicate)
        else
            TC->>ZS: classify(candidate_text, CATEGORY_LABELS)
            ZS-->>TC: {label, score}
            alt score > 0.5
                TC->>TC: assign predicted_category
            else
                TC->>TC: default_category
            end
            TC->>TC: add to filtered_list
        end
    end
    
    TC-->>P: filtered_candidates[with _predicted_category, _category_confidence]
```

---

## Daily Schedule & Slot Allocation

```mermaid
gantt
    title Daily Production Schedule (4 Slots)
    dateFormat  HH:mm
    axisFormat %H:%M
    
    section Slot A (06:00 UTC)
    Trending Fetch     :a1, 06:00, 10m
    Topic Selection    :a2, after a1, 5m
    Script Gen         :a3, after a2, 3m
    Audio Gen          :a4, after a3, 2m
    Visual Fetch       :a5, after a4, 5m
    Video Render       :a6, after a5, 8m
    Upload             :a7, after a6, 3m
    
    section Slot B (12:00 UTC)
    Trending Fetch     :b1, 12:00, 10m
    Topic Selection    :b2, after b1, 5m
    Script Gen         :b3, after b2, 3m
    Audio Gen          :b4, after b3, 2m
    Visual Fetch       :b5, after b4, 5m
    Video Render       :b6, after b5, 8m
    Upload             :b7, after b6, 3m
    
    section Slot C (18:00 UTC) - Longform
    Trending Fetch     :c1, 18:00, 10m
    Topic Selection    :c2, after c1, 5m
    Longform Script    :c3, after c2, 10m
    Audio Gen          :c4, after c3, 5m
    Visual Fetch       :c5, after c4, 10m
    Video Render       :c6, after c5, 30m
    Upload             :c7, after c6, 5m
    
    section Slot D (00:00 UTC)
    Trending Fetch     :d1, 00:00, 10m
    Topic Selection    :d2, after d1, 5m
    Script Gen         :d3, after d2, 3m
    Audio Gen          :d4, after d3, 2m
    Visual Fetch       :d5, after d4, 5m
    Video Render       :d6, after d5, 8m
    Upload             :d7, after d6, 3m
```

---

## Data Flow: Article → Video

```mermaid
flowchart LR
    subgraph "Input"
        A1[Article Title]
        A2[Article Summary]
        A3[Source URL]
        A4[Engagement Metrics]
    end
    
    subgraph "Enrichment"
        E1[Zero-Shot Category]
        E2[Embedding Vector]
        E3[Source Priority Boost]
        E4[Engagement Score 0-100]
    end
    
    subgraph "Script Gen"
        S1[Gemini Prompt]
        S2[Structured JSON]
        S3[subtitle_chunks]
        S4[Visual Prompts]
        S5[Retention Cues]
    end
    
    subgraph "Production"
        P1[Voiceover Audio]
        P2[Word Timestamps]
        P3[Visual Chunks]
        P4[B-Roll Assets]
        P5[Avatar Video]
        P6[Composed Frames]
    end
    
    subgraph "Output"
        O1[MP4 Video]
        O2[Thumbnail]
        O3[Metadata/SEO]
        O4[Upload Receipts]
    end
    
    A1 --> E1
    A2 --> E1
    A3 --> E2
    A4 --> E4
    A1 --> E2
    A2 --> E2
    
    E1 --> S1
    E2 --> S1
    E3 --> S1
    E4 --> S1
    
    S1 --> S2
    S2 --> S3
    S2 --> S4
    S2 --> S5
    
    S3 --> P2
    S4 --> P3
    S4 --> P4
    S5 --> P6
    
    P1 --> P6
    P2 --> P6
    P3 --> P6
    P4 --> P6
    P5 --> P6
    
    P6 --> O1
    S2 --> O2
    S2 --> O3
    O1 --> O4
```

---

## Component Dependency Graph

```mermaid
graph LR
    subgraph "Config & Core"
        CFG[config.py]
        TRK[topic_tracker.py]
        LOG[logging_config.py]
    end
    
    subgraph "Intelligence Layer"
        TC[topic_classifier.py]
        ST[sentence-transformers]
        ZSC[transformers/ZSC]
    end
    
    subgraph "Trending Engine"
        TE[trending_engine.py]
        GH[fetch_github_trending]
        HF[fetch_huggingface]
        MD[fetch_medium_rss]
        AX[fetch_arxiv]
    end
    
    subgraph "Script & Audio"
        GS[gemini_script.py]
        AG[audio_gen.py]
        CB[chunk_builder.py]
    end
    
    subgraph "Visual & Video"
        VF[pexels_fetcher.py]
        NS[nano_scene_gen.py]
        VG[video_gen.py]
        SG[screenshot_gen.py]
        TG[thumbnail_gen.py]
    end
    
    subgraph "Distribution"
        YT[youtube_upload.py]
        IG[instagram_upload.py]
        TT[threads_upload.py]
        FB[facebook_upload.py]
        TL[telegram_selector.py]
    end
    
    CFG --> TE
    CFG --> TC
    CFG --> GS
    CFG --> VG
    
    TRK --> TC
    TRK --> GS
    
    TE --> TC
    TC --> GS
    
    GS --> AG
    GS --> CB
    GS --> VF
    GS --> NS
    
    AG --> VG
    CB --> VG
    VF --> VG
    NS --> VG
    SG --> VG
    
    VG --> TG
    VG --> YT
    VG --> IG
    VG --> TT
    VG --> FB
    VG --> TL
```

---

## Error Handling & Retry Flow

```mermaid
flowchart TD
    Attempt[Attempt N/MAX] --> GenScript[Generate Script]
    GenScript --> ValidScript{Script Valid?}
    
    ValidScript -->|No| RetryScript[Increment Attempt]
    RetryScript --> CheckMax{Attempt >= MAX?}
    CheckMax -->|Yes| Fail[Fail Pipeline]
    CheckMax -->|No| Sleep[Sleep 60s if %3==0]
    Sleep --> Attempt
    
    ValidScript -->|Yes| GenAudio[Generate Audio]
    GenAudio --> ValidAudio{Audio OK?}
    ValidAudio -->|No| RetryScript
    
    ValidAudio --> CaptureSS[Capture Screenshot]
    CaptureSS --> ValidSS{Screenshot OK?}
    ValidSS -->|No| TrackFail[Add to Failed Topics]
    TrackFail --> RetryScript
    
    ValidSS --> FetchVisuals[Fetch Visuals]
    FetchVisuals --> Render[Render Video]
    Render --> ValidVideo{Video OK?}
    ValidVideo -->|No| RetryScript
    
    ValidVideo -->|Yes| Upload[Upload to Platforms]
    Upload --> Done[Success]
```

---

## Retention Engine Visual Pattern

```mermaid
timeline
    title Retention Pattern Interrupts (Every 2-5 seconds)
    
    0s : Hook Overlay\n(3s max)
    2s : Visual Cut\n(Snap-zoom 1.08x)
    5s : Pattern Interrupt\n(Glitch + Flash + SFX)
    7s : Entity Tag\n(Company/Person)
    10s : Visual Cut\n(Whip Pan)
    12s : Screenshot\n(Evidence/GitHub)
    15s : Pattern Interrupt\n(Bass Hit + Glitch)
    17s : Entity Tag
    20s : Visual Cut\n(Zoom Punch)
    25s : CTA Pill\n(Subscribe/Link)
    30s : End Screen\n(Loop/Next Video)
```

---

## Rendering: Viewing Diagrams

These diagrams use **Mermaid.js** syntax. To view them:

1. **VS Code**: Install "Markdown Preview Mermaid Support" extension
2. **GitHub/GitLab**: Native rendering in `.md` files
3. **Obsidian/Notion**: Paste directly
4. **Online**: https://mermaid.live
5. **CLI**: `npx -p @mermaid-js/mermaid-cli mmdc -i ARCHITECTURE.md -o diagrams/`

```bash
# Generate all diagrams as PNG/SVG
npm install -g @mermaid-js/mermaid-cli
mmdc -i docs/ARCHITECTURE.md -o docs/diagrams/ -b transparent
```