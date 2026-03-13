"""
phonetic_dict.py — Comprehensive phonetic correction for F5-TTS voice cloning.

STRATEGY:
1. Static PHONETIC_DICT: 350+ hand-tuned words across 12 categories
2. g2p_en fallback: Auto-detects remaining tricky words using CMU Pronouncing Dictionary + neural G2P
3. Gemini per-script map: Dynamic additions from the AI prompt

The static dict uses hyphenated respellings that sound correct when spoken by F5-TTS.
Subtitles always show original words (handled by restore_original_words in audio_gen.py).
"""

# ══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE PHONETIC DICTIONARY
# ══════════════════════════════════════════════════════════════════════════════

PHONETIC_DICT = {

    # ─── TECH COMPANIES & BRANDS ──────────────────────────────────────────────
    "NVIDIA": "En-vid-ee-uh",
    "AMD": "A-M-D",
    "ASUS": "Ay-soos",
    "Huawei": "Wah-way",
    "Xiaomi": "Shau-mee",
    "OnePlus": "Wun-Plus",
    "Qualcomm": "Kwal-kom",
    "Broadcom": "Brawd-kom",
    "Cisco": "Sis-koh",
    "Adobe": "Uh-doh-bee",
    "Salesforce": "Sails-fors",
    "Palantir": "Pa-lan-teer",
    "Databricks": "Day-tah-bricks",
    "Snowflake": "Snow-flake",
    "MongoDB": "Mong-oh-D-B",
    "Supabase": "Soo-pah-base",
    "Vercel": "Ver-sell",
    "Netlify": "Net-lih-fy",
    "Figma": "Fig-mah",
    "Canva": "Can-vah",
    "Spotify": "Spot-ih-fy",
    "Pinterest": "Pin-ter-est",
    "Twitch": "Twich",
    "TikTok": "Tick-Tok",
    "ByteDance": "Bite-Dance",
    "Baidu": "By-doo",
    "Tencent": "Ten-sent",
    "Alibaba": "Al-ee-bah-bah",
    "Samsung": "Sam-sung",
    "Hyundai": "Hun-day",
    "Porsche": "Por-shuh",
    "Hermes": "Air-mez",
    "Nike": "Ny-kee",
    "Asana": "Ah-sah-nah",
    "Trello": "Trel-oh",
    "Jira": "Jee-ruh",
    "Notion": "Noh-shun",
    "Airtable": "Air-table",

    # ─── AI / MACHINE LEARNING ────────────────────────────────────────────────
    "AI": "A-I",
    "ML": "M-L",
    "NLP": "N-L-P",
    "LLM": "L-L-M",
    "LLMs": "L-L-M-s",
    "GPT": "G-P-T",
    "AGI": "A-G-I",
    "GPU": "G-P-U",
    "GPUs": "G-P-U-s",
    "TPU": "T-P-U",
    "CPU": "C-P-U",
    "CPUs": "C-P-U-s",
    "CUDA": "Koo-dah",
    "VRAM": "Vee-ram",
    "RAM": "Ram",
    "SSD": "S-S-D",
    "API": "A-P-I",
    "APIs": "A-P-I-s",
    "SDK": "S-D-K",
    "CLI": "C-L-I",
    "BERT": "Burt",
    "DALL-E": "Dah-lee",
    "DALL·E": "Dah-lee",
    "GAN": "Gan",
    "GANs": "Gans",
    "RNN": "R-N-N",
    "CNN": "C-N-N",
    "LSTM": "L-S-T-M",
    "LoRA": "Low-rah",
    "QLoRA": "Cue-Low-rah",
    "RLHF": "R-L-H-F",
    "RAG": "Rag",
    "SOTA": "Soh-tah",
    "OpenAI": "Open-A-I",
    "Anthropic": "An-throp-ick",
    "Claude": "Clod",
    "Gemini": "Jem-ih-nye",
    "Mistral": "Miss-trahl",
    "Mixtral": "Mix-trahl",
    "Midjourney": "Mid-jur-nee",
    "Stability": "Stah-bil-ih-tee",
    "StabilityAI": "Stah-bil-ih-tee A-I",
    "Runway": "Run-way",
    "ElevenLabs": "Eleven-Labs",
    "Kokoro": "Koh-koh-roh",
    "DeepSeek": "Deep-Seek",
    "DeepMind": "Deep-Mind",
    "Groq": "Grok",
    "SambaNova": "Sam-ba-Noh-vah",
    "Meta": "Meh-tah",
    "Llama": "Lah-mah",
    "Phi": "Fy",
    "Cohere": "Co-heer",
    "Hugging": "Hug-ging",
    "HuggingFace": "Hug-ging-Face",
    "PyTorch": "Pie-Torch",
    "TensorFlow": "Ten-ser-flow",
    "Keras": "Keh-ras",
    "scikit": "Sy-kit",
    "NumPy": "Num-pie",
    "SciPy": "Sy-pie",
    "Matplotlib": "Mat-plot-lib",
    "Jupyter": "Joo-pih-ter",
    "diffusion": "dih-few-zhun",
    "transformer": "trans-for-mer",
    "transformers": "trans-for-mers",
    "tokenizer": "toh-ken-eye-zer",
    "tokenization": "toh-ken-eye-zay-shun",
    "embeddings": "em-bed-dings",
    "multimodal": "mul-tee-moh-dul",
    "reinforcement": "ree-in-fors-ment",
    "generative": "jen-er-uh-tiv",
    "discriminative": "dis-krim-in-uh-tiv",
    "autonomous": "aw-ton-uh-mus",
    "autoregressive": "aw-toh-ree-gres-siv",
    "latent": "lay-tent",
    "inference": "in-fer-ents",
    "inferences": "in-fer-ent-sez",
    "parameters": "puh-ram-uh-ters",
    "hyperparameters": "hy-per-puh-ram-uh-ters",
    "fine-tuning": "fine-too-ning",
    "fine-tuned": "fine-toond",
    "pre-trained": "pree-traind",
    "dataset": "day-tah-set",
    "datasets": "day-tah-sets",
    "benchmark": "bench-mark",
    "benchmarks": "bench-marks",
    "quantization": "kwan-tih-zay-shun",
    "quantized": "kwan-tized",
    "pruning": "proo-ning",
    "hallucination": "huh-loo-sih-nay-shun",
    "hallucinations": "huh-loo-sih-nay-shuns",
    "stochastic": "stoh-kas-tick",
    "deterministic": "dee-ter-min-is-tick",
    "perceptron": "per-sep-tron",
    "backpropagation": "back-prop-uh-gay-shun",
    "convolutional": "con-voh-loo-shun-ul",
    "recurrent": "ree-kur-ent",
    "architecture": "ar-kih-tek-chur",
    "architectures": "ar-kih-tek-churs",
    "epoch": "ee-pok",
    "epochs": "ee-poks",
    "gradient": "gray-dee-ent",
    "gradients": "gray-dee-ents",
    "overfitting": "oh-ver-fit-ting",
    "underfitting": "un-der-fit-ting",
    "regularization": "reg-yoo-lur-eye-zay-shun",
    "neural": "nur-ul",
    "neurons": "nur-ons",
    "synapse": "sin-aps",
    "cognitive": "kog-nih-tiv",
    "heuristic": "hew-ris-tick",
    "latency": "lay-ten-see",
    "scaling": "skay-ling",
    "throughput": "thru-put",
    "quantize": "kwan-tize",
    "tokenize": "toh-ken-eye-ze",
    "decoder": "dee-koh-der",
    "encoder": "en-koh-der",
    "recursive": "ree-kur-siv",
    "iterative": "it-er-uh-tiv",
    "substrate": "sub-strayt",
    "lithography": "ih-thog-ruh-fee",
    "microprocessor": "my-kroh-prah-ses-er",

    # ─── SOFTWARE / DEVTOOLS ──────────────────────────────────────────────────
    "JavaScript": "Jah-va-skript",
    "TypeScript": "Type-skript",
    "Python": "Py-thon",
    "Rust": "Rust",
    "Golang": "Go-lang",
    "Kotlin": "Kot-lin",
    "Swift": "Swift",
    "Svelte": "Svelt",
    "Vue": "View",
    "NGINX": "Engine-X",
    "SQL": "See-kw-uhl",
    "MySQL": "My-See-kw-uhl",
    "PostgreSQL": "Post-gres",
    "SQLite": "S-Q-Lite",
    "Redis": "Red-iss",
    "Kubernetes": "Koo-ber-net-eez",
    "Docker": "Dok-er",
    "Terraform": "Tehr-uh-form",
    "Ansible": "An-sih-bul",
    "Pexels": "Pex-uls",
    "GitHub": "Git-hub",
    "GitLab": "Git-lab",
    "Linux": "Lin-icks",
    "Ubuntu": "Oo-boon-too",
    "Debian": "Deb-ee-un",
    "Fedora": "Feh-dor-uh",
    "macOS": "Mac-O-S",
    "iOS": "Eye-O-S",
    "GUI": "Goo-ee",
    "OAuth": "Oh-auth",
    "regex": "rej-ex",
    "cache": "kash",
    "caching": "kash-ing",
    "daemon": "dee-mun",
    "mutex": "mew-tex",
    "sudo": "soo-doo",
    "wget": "w-get",
    "cURL": "curl",
    "LaTeX": "Lah-tek",
    "IEEE": "Eye-triple-E",
    "ASCII": "As-kee",
    "UTF": "U-T-F",
    "JSON": "Jay-son",
    "YAML": "Yam-ul",
    "TOML": "Tom-ul",
    "HTTP": "H-T-T-P",
    "HTTPS": "H-T-T-P-S",
    "SSH": "S-S-H",
    "FTP": "F-T-P",
    "URL": "U-R-L",
    "URLs": "U-R-L-s",
    "DNS": "D-N-S",
    "CDN": "C-D-N",
    "SSL": "S-S-L",
    "TLS": "T-L-S",
    "IPv4": "I-P-version-4",
    "IPv6": "I-P-version-6",
    "WebSocket": "Web-Sock-et",
    "OAuth2": "Oh-auth-two",
    "JWT": "J-W-T",
    "SaaS": "Sass",
    "PaaS": "Pass",
    "IaaS": "Eye-as",
    "DevOps": "Dev-Ops",
    "CI/CD": "C-I-C-D",
    "CICD": "C-I-C-D",

    # ─── NUMBERS & SCALES ─────────────────────────────────────────────────────
    "million": "mil-yun",
    "millions": "mil-yuns",
    "billion": "bil-yun",
    "billions": "bil-yuns",
    "trillion": "tril-yun",
    "trillions": "tril-yuns",
    "quadrillion": "kwad-ril-yun",
    "gigabyte": "gig-uh-bite",
    "gigabytes": "gig-uh-bites",
    "megabyte": "meg-uh-bite",
    "megabytes": "meg-uh-bites",
    "terabyte": "terr-uh-bite",
    "terabytes": "terr-uh-bites",
    "petabyte": "pet-uh-bite",
    "petabytes": "pet-uh-bites",
    "gigahertz": "gig-uh-hurts",
    "megahertz": "meg-uh-hurts",
    "percentage": "per-sent-ij",
    "percentages": "per-sent-ij-ez",

    # ─── SILENT LETTERS & SPELLING TRAPS ──────────────────────────────────────
    "island": "eye-land",
    "islands": "eye-lands",
    "colonel": "ker-nel",
    "subtle": "sut-ul",
    "subtly": "sut-lee",
    "debt": "det",
    "doubt": "dowt",
    "receipt": "reh-seet",
    "salmon": "sam-un",
    "almond": "ah-mund",
    "Wednesday": "Wenz-day",
    "February": "Feb-roo-air-ee",
    "comfortable": "kum-fert-uh-bul",
    "vegetable": "vej-tuh-bul",
    "chocolate": "chok-lit",
    "interesting": "in-trest-ing",
    "different": "dif-rent",
    "temperature": "tem-pruh-chur",
    "basically": "bay-sik-lee",
    "probably": "prob-ub-lee",
    "actually": "ak-choo-uh-lee",
    "naturally": "natch-ruh-lee",
    "generally": "jen-ruh-lee",
    "especially": "eh-spesh-uh-lee",
    "restaurant": "rest-rahnt",
    "necessary": "ness-uh-sair-ee",
    "laboratory": "lab-ruh-tor-ee",
    "category": "kat-uh-gor-ee",
    "definitely": "def-in-it-lee",
    "literally": "lit-er-uh-lee",
    "particularly": "par-tik-yoo-lar-lee",
    "temporarily": "tem-puh-rair-uh-lee",
    "extraordinary": "ek-stror-din-air-ee",
    "simultaneously": "sy-mul-tay-nee-us-lee",
    "hierarchy": "hy-rahr-kee",
    "paradigm": "par-uh-dime",
    "paradigms": "par-uh-dimes",
    "often": "off-en",
    "listen": "lis-en",
    "mortgage": "mor-gij",
    "pneumonia": "new-moh-nee-uh",
    "psychology": "sy-kol-uh-jee",
    "psychiatry": "sy-ky-uh-tree",
    "pseudocode": "soo-doh-code",
    "gnu": "new",
    "gnome": "nome",
    "gauge": "gayj",
    "height": "hite",
    "draught": "draft",
    "suite": "sweet",
    "bouquet": "boo-kay",
    "quay": "kee",
    "choir": "kwire",

    "knight": "nite",
    "knowledge": "nol-ij",
    "mnemonic": "neh-mon-ick",
    "rhetoric": "ret-or-ick",
    "rhythm": "rith-um",
    "algorithm": "al-go-rith-um",
    "algorithms": "al-go-rith-ums",

    # ─── HETERONYMS (Same spelling, different pronunciation by meaning) ───────
    "read": "red",            # Past tense context
    "lead": "led",            # Metal context
    "live": "liv",            # Adjective context
    "content": "con-tent",    # Adjective (satisfied) context
    "present": "prez-ent",    # Noun context
    "record": "rek-ord",      # Noun context
    "object": "ob-jekt",      # Noun context
    "produce": "prod-oos",    # Noun context
    "project": "proj-ekt",    # Noun context
    "estimate": "es-tih-mut", # Noun context
    "conduct": "kon-dukt",    # Noun context

    # ─── COMMONLY MISPRONOUNCED ENGLISH ───────────────────────────────────────
    "says": "sez",
    "said": "sed",
    "iron": "i-ern",
    "chaos": "kay-oss",
    "queue": "kyew",
    "recipe": "res-ih-pee",
    "debacle": "duh-bah-kul",
    "epitome": "ih-pit-uh-mee",
    "hyperbole": "hy-per-buh-lee",
    "meme": "meem",
    "niche": "neesh",
    "genre": "zhon-ruh",
    "rendezvous": "ron-day-voo",
    "entrepreneur": "on-truh-pruh-ner",
    "entrepreneurs": "on-truh-pruh-ners",
    "ubiquitous": "yoo-bik-wih-tus",
    "phenomenon": "fuh-nom-uh-non",
    "phenomena": "fuh-nom-uh-nah",
    "anonymity": "an-uh-nim-ih-tee",
    "anonymous": "uh-non-uh-mus",
    "vulnerability": "vul-ner-uh-bil-ih-tee",
    "vulnerabilities": "vul-ner-uh-bil-ih-teez",
    "instantaneously": "in-stan-tay-nee-us-lee",
    "instantaneous": "in-stan-tay-nee-us",
    "instantly": "in-stunt-ly",
    "only": "own-lee",
    "incredible": "in-cred-uh-bul",
    "incredibly": "in-cred-uh-blee",
    "accessible": "ak-ses-uh-bul",
    "accessibility": "ak-ses-uh-bil-ih-tee",
    "miscellaneous": "mis-uh-lay-nee-us",
    "mischievous": "mis-chuh-vus",
    "specific": "spuh-sif-ick",
    "specifically": "spuh-sif-ick-lee",
    "pronunciation": "pruh-nun-see-ay-shun",
    "debris": "duh-bree",
    "elite": "eh-leet",
    "regime": "reh-zheem",
    "faux": "foh",
    "coup": "koo",
    "cache": "kash",
    "facade": "fuh-sahd",
    "fiancé": "fee-on-say",
    "cliché": "klee-shay",
    "résumé": "rez-oo-may",
    "naive": "ny-eev",
    "mature": "muh-choor",
    "premiere": "preh-meer",
    "prestige": "preh-steezh",
    "prestigious": "preh-stee-jus",
    "espionage": "es-pee-uh-nahzh",
    "sabotage": "sab-uh-tahzh",
    "camouflage": "kam-uh-flahzh",
    "entourage": "on-too-rahzh",
    "montage": "mon-tahzh",
    "massage": "muh-sahzh",
    "mirage": "mih-rahzh",
    "garage": "guh-rahzh",
    "jewelry": "jool-ree",
    "library": "ly-brair-ee",
    "potentially": "poh-ten-shully",
    "relevant": "rel-uh-vunt",
    "prioritize": "pry-or-ih-tize",
    "analysis": "uh-nal-ih-sis",
    "strategy": "strat-uh-jee",
    "development": "deh-vel-up-ment",
    "industry": "in-dus-tree",
    "innovation": "in-noh-vay-shun",
    "successfully": "suk-sess-ful-lee",
    "significantly": "sig-nif-ih-kunt-lee",
    "essentially": "eh-sen-shully",
    "availability": "uh-vay-luh-bil-ih-tee",
    "environment": "en-vy-run-ment",
    "government": "guv-ern-ment",
    "experience": "ek-speer-ee-ents",
    "performance": "per-for-munts",
    "efficiency": "ih-fish-un-see",
    "effective": "ih-fek-tiv",
    "effectively": "ih-fek-tiv-lee",
    "opportunity": "op-er-too-nih-tee",
    "significant": "sig-nif-ih-kunt",
    "society": "suh-sy-ih-tee",
    "technology": "tek-nol-uh-jee",
    "technologies": "tek-nol-uh-jeez",
    "artificial": "ar-tih-fish-ul",
    "intelligence": "in-tel-ih-junts",
    "unbelievable": "un-bee-leev-uh-bul",
    "momentous": "moh-men-tus",
    "pioneer": "py-uh-neer",
    "commercial": "kuh-mer-shul",
    "financial": "fih-nan-shul",
    "economic": "ee-koh-nom-ick",
    "political": "puh-lit-ih-kul",
    "scientific": "sy-en-tif-ick",
    "theoretical": "thee-oh-ret-ih-kul",
    "practical": "prak-tih-kul",
    "technical": "tek-nih-kul",


    "collage": "kuh-lahzh",
    "leverage": "lev-er-ij",
    "average": "av-er-ij",
    "coverage": "kuv-er-ij",
    "mortgage": "mor-gij",

    # ─── SCIENCE & ACADEMIA ───────────────────────────────────────────────────
    "nuclear": "new-klee-er",
    "molecular": "muh-lek-yoo-ler",
    "pharmaceutical": "far-muh-soo-tih-kul",
    "genome": "jee-nome",
    "genomics": "jeh-nom-icks",
    "photosynthesis": "foh-toh-sin-thuh-sis",
    "thermodynamics": "thur-moh-dy-nam-icks",
    "electromagnetic": "ee-lek-troh-mag-net-ick",
    "cryptocurrency": "krip-toh-kur-en-see",
    "cryptocurrencies": "krip-toh-kur-en-seez",
    "blockchain": "block-chain",
    "decentralized": "dee-sen-truh-lized",
    "semiconductor": "sem-ee-con-duk-ter",
    "semiconductors": "sem-ee-con-duk-ters",
    "nanotechnology": "nan-oh-tek-nol-uh-jee",
    "biotechnology": "by-oh-tek-nol-uh-jee",
    "cybersecurity": "sy-ber-suh-kyoor-ih-tee",
    "infrastructure": "in-fruh-struk-chur",
    "infrastructure's": "in-fruh-struk-churs",
    "scalability": "skay-luh-bil-ih-tee",
    "sustainability": "suh-stay-nuh-bil-ih-tee",
    "interoperability": "in-ter-op-er-uh-bil-ih-tee",
    "semiconductor": "sem-ee-con-duk-ter",

    # ─── BUSINESS / FINANCE ───────────────────────────────────────────────────
    "revenue": "rev-uh-new",
    "revenues": "rev-uh-news",
    "acquisition": "ak-wih-zih-shun",
    "acquisitions": "ak-wih-zih-shuns",
    "valuation": "val-yoo-ay-shun",
    "equity": "ek-wih-tee",
    "portfolio": "port-foh-lee-oh",
    "dividend": "div-ih-dend",
    "fiduciary": "fih-doo-shee-air-ee",
    "amortization": "am-or-tih-zay-shun",
    "depreciation": "deh-pree-shee-ay-shun",

    # ─── COMMON SUFFIXES THAT TRIP UP TTS ─────────────────────────────────────
    "breakthrough": "break-thrue",
    "breakthroughs": "break-thrues",
    "capabilities": "kay-puh-bil-uh-teez",
    "capability": "kay-puh-bil-uh-tee",
    "reliability": "ree-ly-uh-bil-ih-tee",
    "compatibility": "kum-pat-uh-bil-ih-tee",
    "functionality": "funk-shun-al-ih-tee",
    "revolutionary": "rev-uh-loo-shun-air-ee",
    "unprecedented": "un-pres-uh-dent-ed",
    "comprehensive": "kom-pree-hen-siv",
    "approximately": "uh-proks-ih-mit-lee",
    "implementation": "im-pluh-men-tay-shun",
    "implementations": "im-pluh-men-tay-shuns",
    "manufacturing": "man-yoo-fak-chur-ing",
    "communication": "kuh-mew-nih-kay-shun",
    "communications": "kuh-mew-nih-kay-shuns",
    "configuration": "kon-fig-yoo-ray-shun",
    "documentation": "dok-yoo-men-tay-shun",
    "collaboration": "kuh-lab-or-ay-shun",
    "optimization": "op-tih-my-zay-shun",
    "visualization": "vizh-oo-ul-eye-zay-shun",
    "experimentation": "ek-sper-ih-men-tay-shun",
    "democratization": "deh-mok-ruh-tih-zay-shun",

    # ─── ARCHIVES / MISC ─────────────────────────────────────────────────────
    "archives": "ar-kives",
    "archive": "ar-kive",
    "minutiae": "mih-noo-shee-ee",
    "albeit": "awl-bee-it",
    "segue": "seg-way",
    "lingerie": "lon-zhur-ay",
    "connoisseur": "kon-uh-sur",
    "etiquette": "et-ih-ket",
    "bureaucracy": "byoo-rok-ruh-see",
    "sovereignty": "sov-rin-tee",
    "plethora": "pleth-or-uh",
    "analogous": "uh-nal-uh-gus",
    "superfluous": "soo-per-floo-us",
    "ambiguous": "am-big-yoo-us",
    "ubiquitous": "yoo-bik-wih-tus",
    "prerogative": "preh-rog-uh-tiv",
    "deteriorate": "dih-teer-ee-or-ate",
    "deterioration": "dih-teer-ee-or-ay-shun",
}


# ══════════════════════════════════════════════════════════════════════════════
# AUTOMATIC PRONUNCIATION DETECTION (g2p_en fallback)
# ══════════════════════════════════════════════════════════════════════════════

_g2p = None

def _get_g2p():
    """Lazy-load g2p_en to avoid import overhead."""
    global _g2p
    if _g2p is None:
        try:
            from g2p_en import G2p
            _g2p = G2p()
            print("✓ g2p_en loaded for automatic pronunciation correction.")
        except ImportError:
            print("⚠ g2p_en not installed. Only using static phonetic dict.")
            _g2p = False  # Mark as attempted but failed
    return _g2p


# Words we know are fine — skip auto-correction for these
_SAFE_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "so", "than", "too", "very", "just",
    "about", "also", "but", "and", "or", "if", "that", "this",
    "these", "those", "what", "which", "who", "whom", "it", "its",
    "he", "she", "they", "them", "his", "her", "my", "your", "our",
    "their", "we", "you", "me", "him", "us", "up", "down", "new",
    "old", "now", "right", "left", "big", "small", "long", "short",
    "good", "bad", "well", "much", "many", "still", "back", "even",
    "get", "got", "go", "went", "come", "came", "make", "made",
    "take", "took", "see", "saw", "know", "knew", "think", "thought",
    "say", "tell", "told", "give", "gave", "find", "found", "put",
    "keep", "kept", "let", "want", "need", "use", "used", "try",
    "tried", "work", "run", "look", "like", "time", "way", "day",
    "man", "men", "part", "place", "case", "week", "hand", "point",
    "help", "turn", "start", "show", "hear", "play", "move",
    "set", "end", "far", "real", "full", "life", "last", "next",
    "same", "able", "data", "tech", "code", "app", "web", "net",
    "yes", "two", "three", "four", "five", "six", "ten", "one",
}

# ARPAbet vowel to readable respelling mapping
_ARPA_TO_READABLE = {
    "AA": "ah", "AE": "a", "AH": "uh", "AO": "aw",
    "AW": "ow", "AY": "eye", "EH": "eh", "ER": "er",
    "EY": "ay", "IH": "ih", "IY": "ee", "OW": "oh",
    "OY": "oy", "UH": "oo", "UW": "oo",
}

_ARPA_CONSONANTS = {
    "B": "b", "CH": "ch", "D": "d", "DH": "th", "F": "f",
    "G": "g", "HH": "h", "JH": "j", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ng", "P": "p", "R": "r",
    "S": "s", "SH": "sh", "T": "t", "TH": "th", "V": "v",
    "W": "w", "Y": "y", "Z": "z", "ZH": "zh",
}


def _phonemes_to_respelling(phonemes):
    """Convert ARPAbet phonemes to a hyphenated human-readable respelling."""
    syllables = []
    current = ""
    for p in phonemes:
        # Strip stress markers (0, 1, 2)
        clean = ''.join(c for c in p if not c.isdigit())
        if clean in _ARPA_TO_READABLE:
            # Vowel — ends current syllable
            current += _ARPA_TO_READABLE[clean]
            syllables.append(current)
            current = ""
        elif clean in _ARPA_CONSONANTS:
            current += _ARPA_CONSONANTS[clean]
        # else: skip punctuation/space tokens
    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)
    return "-".join(syllables) if syllables else None


def auto_detect_hard_words(text):
    """
    Uses g2p_en to detect words that are likely to be mispronounced.
    Returns a dict of {word: phonetic_respelling} for words that need help.
    
    Only detects words that:
    1. Are NOT in the static PHONETIC_DICT already
    2. Are NOT in the safe-words list
    3. Have 3+ syllables OR contain unusual letter patterns
    """
    import re
    g2p = _get_g2p()
    if not g2p:
        return {}

    words = re.findall(r"[A-Za-z'-]+", text)
    auto_map = {}

    for word in words:
        lower = word.lower().strip("'-")
        if not lower or len(lower) < 4:
            continue
        if lower in _SAFE_WORDS:
            continue
        if lower in PHONETIC_DICT or word in PHONETIC_DICT:
            continue

        try:
            phonemes = g2p(lower)
            # Filter to only actual phonemes (not spaces/punctuation)
            real_phonemes = [p for p in phonemes if p.strip() and not p.isspace()]
            
            # Count vowel phonemes = approximate syllable count
            vowel_count = sum(1 for p in real_phonemes
                            if ''.join(c for c in p if not c.isdigit()) in _ARPA_TO_READABLE)

            # Only auto-correct words with 2+ syllables (changed from 3 to catch more tech terms)
            if vowel_count >= 2:
                respelling = _phonemes_to_respelling(real_phonemes)
                if respelling and respelling != lower:
                    auto_map[word] = respelling
        except Exception:
            pass

    if auto_map:
        print(f"🔤 Auto-detected {len(auto_map)} pronunciation corrections: {list(auto_map.keys())[:10]}...")
    return auto_map
