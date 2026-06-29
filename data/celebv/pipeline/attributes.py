"""
CelebV-HQ appearance-attribute helpers (shared by stage1 filter.py + stage3 main.py).

CelebV-HQ stores per-clip `attributes.appearance` as a 40-d 0/1 vector. The index
-> name mapping (`meta_info.appearance_mapping`) is a DATASET CONSTANT, mirrored
here as APPEARANCE_MAPPING so both the candidate filter (selection) and the 
per-clip pipeline (caption synthesis) decode it identically.

Usage:
  - filter.py  : hair-color bucketing + clip exclusion (stage 1).
  - main.py    : build_caption() -> the structured caption written to meta.json
                 (stage 3); the empty-caption CelebV clips thus gain a weak,
                 attribute-grounded text condition that mentions hair explicitly.

Special design notes:
  - Hair COLOR is the balancing axis: 4 explicit colors + an "other" bucket
    (clips with hair but no labelled colour) = 5 buckets total.
  - Hair MORPHOLOGY (straight/wavy/long) is descriptive only -> goes into the
    caption + distribution report, never used to select/balance.
  - Excluded clips (never enter candidate.json):
      * wearing_hat -> hair is occluded by a hat.
      * bald        -> no hair region for SCHP to segment (useless for hair edit).
"""
# TODO: WAN2.1的技术报告提到了关于训练和推理时prompt分布一致的说明
# 所以后期可以让QwenVL-8B根据如下关键词随机组成caption模拟用户的prompt


from __future__ import annotations
from typing import List, Optional, Sequence

# 40-d appearance vector index -> attribute name (CelebV-HQ meta_info.appearance_mapping).
APPEARANCE_MAPPING: List[str] = [
    "blurry",            # 0
    "male",              # 1
    "young",             # 2
    "chubby",            # 3
    "pale_skin",         # 4
    "rosy_cheeks",       # 5
    "oval_face",         # 6
    "receding_hairline", # 7
    "bald",              # 8
    "bangs",             # 9
    "black_hair",        # 10
    "blonde_hair",       # 11
    "gray_hair",         # 12
    "brown_hair",        # 13
    "straight_hair",     # 14
    "wavy_hair",         # 15
    "long_hair",         # 16
    "arched_eyebrows",   # 17
    "bushy_eyebrows",    # 18
    "bags_under_eyes",   # 19
    "eyeglasses",        # 20
    "sunglasses",        # 21
    "narrow_eyes",       # 22
    "big_nose",          # 23
    "pointy_nose",       # 24
    "high_cheekbones",   # 25
    "big_lips",          # 26
    "double_chin",       # 27
    "no_beard",          # 28
    "5_o_clock_shadow",  # 29
    "goatee",            # 30
    "mustache",          # 31
    "sideburns",         # 32
    "heavy_makeup",      # 33
    "wearing_earrings",  # 34
    "wearing_hat",       # 35
    "wearing_lipstick",  # 36
    "wearing_necklace",  # 37
    "wearing_necktie",   # 38
    "wearing_mask",      # 39
]

IDX = {name: i for i, name in enumerate(APPEARANCE_MAPPING)}

# Hair-color balancing axis: 4 explicit colours + a catch-all "other" bucket.
HAIR_COLOR_NAMES: List[str] = ["black_hair", "blonde_hair", "gray_hair", "brown_hair"]
OTHER_BUCKET = "other"
HAIR_BUCKETS: List[str] = HAIR_COLOR_NAMES + [OTHER_BUCKET]      # 5 buckets

# Hair morphology: descriptive only (reported + captioned, never balanced on).
HAIR_MORPH_NAMES: List[str] = ["straight_hair", "wavy_hair", "long_hair"]

# Clips carrying any of these are dropped at stage 1 (no usable hair region).
EXCLUDE_NAMES: List[str] = ["wearing_hat", "bald"]

# 35-d action vector index -> verb (CelebV-HQ meta_info.action_mapping).
ACTION_MAPPING: List[str] = [
    "blow", "chew", "close_eyes", "cough", "cry", "drink", "eat", "frown", "gaze",
    "glare", "head_wagging", "kiss", "laugh", "listen_to_music", "look_around",
    "make_a_face", "nod", "play_instrument", "read", "shake_head", "shout", "sigh",
    "sing", "sleep", "smile", "smoke", "sneer", "sneeze", "sniff", "talk", "turn",
    "weep", "whisper", "wink", "yawn",
]

# Natural present-continuous phrasing per action (index-aligned with ACTION_MAPPING),
ACTION_GERUND: List[str] = [
    "blowing", "chewing", "closing eyes", "coughing", "crying", "drinking", "eating",
    "frowning", "gazing", "glaring", "wagging head", "kissing", "laughing",
    "listening to music", "looking around", "making a face", "nodding",
    "playing an instrument", "reading", "shaking head", "shouting", "sighing",
    "singing", "sleeping", "smiling", "smoking", "sneering", "sneezing", "sniffing",
    "talking", "turning", "weeping", "whispering", "winking", "yawning",
]

# Caption assembly groups (0/1 -> keep when 1).
_ADJ_IDX = [IDX["young"], IDX["chubby"]]                          # adjectives before gender
_WEARING_IDX = [IDX[n] for n in (                                 # "wearing X, Y and Z"
    "wearing_earrings", "wearing_lipstick", "wearing_necklace",
    "wearing_necktie", "wearing_mask",
)]
# Noun phrases for the "with ..." tail: indices 4..33 (pale_skin..heavy_makeup),
# minus 'bald' (never described). Hair colour + morphology live in here, so the
# caption always mentions hair explicitly when labelled.
_NOUN_IDX = [i for i in range(IDX["pale_skin"], IDX["heavy_makeup"] + 1) if i != IDX["bald"]]


def _on(appearance: Sequence[int], i: int) -> bool:
    """Safe 0/1 read."""
    return 0 <= i < len(appearance) and bool(appearance[i])


def is_excluded(appearance: Sequence[int]) -> bool:
    """True if the clip must NOT enter candidate.json (hat-occluded or bald)."""
    return any(_on(appearance, IDX[n]) for n in EXCLUDE_NAMES)


def colors_present(appearance: Sequence[int]) -> List[str]:
    """Explicit hair-colour labels set on this clip (subset of HAIR_COLOR_NAMES)."""
    # returns corresponding color string name (e.g. "black_hair", "blonde_hair", "gray_hair", "brown_hair")
    return [c for c in HAIR_COLOR_NAMES if _on(appearance, IDX[c])]


def actions_present(action: Sequence[int]) -> List[str]:
    """Action verbs set on this clip (subset of ACTION_MAPPING), raw names."""
    return [ACTION_MAPPING[i] for i in range(len(ACTION_MAPPING)) if _on(action, i)]


def morphology_present(appearance: Sequence[int]) -> List[str]:
    """Hair-morphology labels set on this clip (subset of HAIR_MORPH_NAMES)."""
    return [m for m in HAIR_MORPH_NAMES if _on(appearance, IDX[m])]


def _join_and(items: List[str]) -> str:
    """['a'] -> 'a'; ['a','b'] -> 'a and b'; ['a','b','c'] -> 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def build_caption(appearance: Sequence[int], action: Optional[Sequence[int]] = None) -> str:
    """
    Compose a deterministic, attribute-grounded rough caption for a single-person clip:

        a [young] [chubby] {male|female} [wearing X, Y and Z] with N1, N2, and Nk
        [is V1, V2, and Vk].

    Rules (per project spec):
      - blurry            : ignored (CelebV clips are reliably sharp).
      - male              : 0 -> "female", 1 -> "male" (gender always emitted).
      - young / chubby    : adjective, emitted only when 1 (no clean antonym).
      - pale_skin..heavy_makeup : noun phrases, emitted only when 1, EXCEPT 'bald'
                            which is never described (and is filtered upstream anyway).
      - wearing_*         : state phrases, emitted only when 1. 'wearing_hat' is
                            filtered upstream so it never appears here.
      - hair colour + morphology sit in the "with ..." tail, so hair is always
        described when labelled (the signal we care about most for hair editing).
      - action            : optional 35-d vector; set verbs become an "is <gerund(s)>"
                            tail so the caption reads like a natural video description.

    This is a weak structured caption (not grammatically perfect), used as an optional text condition.
    """
    gender = "male" if _on(appearance, IDX["male"]) else "female"
    adjs = [APPEARANCE_MAPPING[i] for i in _ADJ_IDX if _on(appearance, i)]
    subject = "a " + " ".join(adjs + [gender])

    wearing = [
        APPEARANCE_MAPPING[i].replace("wearing_", "").replace("_", " ")
        for i in _WEARING_IDX if _on(appearance, i)
    ]
    nouns = [
        APPEARANCE_MAPPING[i].replace("_", " ")
        for i in _NOUN_IDX if _on(appearance, i)
    ]
    gerunds = (
        [ACTION_GERUND[i] for i in range(len(ACTION_MAPPING)) if _on(action, i)]
        if action else []
    )

    caption = subject
    if wearing:
        caption += " wearing " + _join_and(wearing)
    if nouns:
        caption += " with " + _join_and(nouns)
    if gerunds:
        caption += " is " + _join_and(gerunds)
    return caption + "."
