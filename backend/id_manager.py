"""
id_manager.py  — v4 (Architecture Cleaned)
========================================
Stable official ID assignment with:
  1. Stability gate: track must appear in 18 of last 25 frames.
  2. Appearance-based Re-identification: Trajectory-aware + Zero Position Bias.
  3. Locked mapping: track_id -> official_id is persistent.
  4. Garbage Collection: Prunes stale metadata (>1500 frames) for long-duration matches.
"""

from collections import defaultdict, deque
from typing import Dict, Optional, Tuple, List
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────
STABILITY_WINDOW    = 25
STABILITY_THRESHOLD = 18
MAX_PER_TEAM        = 7

# Re-ID config
REID_GRACE_FRAMES = 120    # frames to keep a lost track eligible for re-ID
REID_DIST_THRESH  = 150    # pixels
REID_APP_THRESH   = 0.75   # cosine sim
REID_STRICT_APP   = 0.82   # pure appearance threshold
STALE_THRESHOLD   = 1500   # 60s @ 25fps

# ── State ──────────────────────────────────────────────────────────────────
_locked:      Dict[int, str] = {}   # track_id -> official_id
_team_locked: Dict[int, int] = {}   # track_id -> team (0 or 1)
_team_counters: Dict[int, int] = {0: 1, 1: 1}

_presence:      Dict[int, deque] = defaultdict(lambda: deque(maxlen=STABILITY_WINDOW))
_team_snapshot: Dict[int, int]   = {}

# Persistent metadata for ReID
_last_box:       Dict[int, Tuple[int, int, int, int]] = {}
_emb_history:    Dict[int, deque]                      = defaultdict(lambda: deque(maxlen=30))
_vel_history:    Dict[int, deque]                      = defaultdict(lambda: deque(maxlen=10))
_last_frame:     Dict[int, int]                        = {}
_current_frame: int = 0

# ── Helpers ────────────────────────────────────────────────────────────────

def _cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-6))

def _get_mean_emb(tid: int) -> Optional[np.ndarray]:
    if tid not in _emb_history or not _emb_history[tid]: return None
    mean_emb = np.mean(_emb_history[tid], axis=0)
    return mean_emb / (np.linalg.norm(mean_emb) + 1e-6)

def _center_dist(box1, box2):
    c1 = ((box1[0]+box1[2])/2, (box1[1]+box1[3])/2)
    c2 = ((box2[0]+box2[2])/2, (box2[1]+box2[3])/2)
    return np.hypot(c1[0]-c2[0], c1[1]-c2[1])

# ── Implementation ─────────────────────────────────────────────────────────

def record_frame(track_boxes: Dict[int, Tuple], team_labels: Dict[int, int]):
    """Main entry point per frame. Processes all visible tracks."""
    global _current_frame
    _current_frame += 1
    
    if _current_frame % 100 == 0:
        cleanup_stale_ids()

    visible_set = set(track_boxes.keys())
    all_known = set(_presence.keys()) | visible_set

    for tid in all_known:
        _presence[tid].append(1 if tid in visible_set else 0)

    for tid, box_data in track_boxes.items():
        if tid in team_labels:
            _team_snapshot[tid] = team_labels[tid]
            
        box = box_data[:4]
        emb = box_data[5] # Expecting (x1,y1,x2,y2,conf,emb)
        
        # 1. Update existing locked track
        if tid in _locked:
            _update_track_state(tid, box, emb)
            continue
            
        # 2. Try to re-identify if it's a new track (presence=1)
        if sum(_presence[tid]) == 1:
            best_tid = _find_reid_match(tid, box, emb, visible_set)
            if best_tid is not None:
                _inherit_id(tid, best_tid)
                _update_track_state(tid, box, emb)
                continue
                
        # 3. Just record as transient track
        _update_track_state(tid, box, emb)

def _update_track_state(tid, box, emb):
    if tid in _last_box:
        c = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        pc = ((_last_box[tid][0]+_last_box[tid][2])/2, (_last_box[tid][1]+_last_box[tid][3])/2)
        _vel_history[tid].append(np.array([c[0]-pc[0], c[1]-pc[1]]))
        
    _last_box[tid] = box
    _emb_history[tid].append(emb)
    _last_frame[tid] = _current_frame

def _find_reid_match(tid, box, emb, visible_set):
    best_tid, best_score = None, -1.0
    
    for old_tid in list(_locked.keys()):
        if old_tid in visible_set: continue
        
        age = _current_frame - _last_frame.get(old_tid, 0)
        if age > REID_GRACE_FRAMES: continue
            
        old_emb = _get_mean_emb(old_tid)
        old_box = _last_box.get(old_tid)
        if old_emb is None or old_box is None: continue
            
        app_sim = _cosine_sim(emb, old_emb)
        dist = _center_dist(box, old_box)
        
        # Motion Similarity
        motion_sim = 1.0
        if tid in _last_box:
            c = ((box[0]+box[2])/2, (box[1]+box[3])/2)
            pc = ((_last_box[tid][0]+_last_box[tid][2])/2, (_last_box[tid][1]+_last_box[tid][3])/2)
            curr_vec = np.array([c[0]-pc[0], c[1]-pc[1]])
            if old_tid in _vel_history and _vel_history[old_tid]:
                hist_vec = np.mean(_vel_history[old_tid], axis=0)
                nv, nh = np.linalg.norm(curr_vec), np.linalg.norm(hist_vec)
                if nv > 5 and nh > 5:
                    motion_sim = np.dot(curr_vec, hist_vec) / (nv * nh + 1e-6)

        is_match = False
        score = 0.0
        
        if app_sim >= REID_STRICT_APP:
            is_match, score = True, app_sim * 4.0
        elif app_sim >= REID_APP_THRESH and dist <= REID_DIST_THRESH * 1.5:
            is_match, score = True, app_sim * 2.0 * (1.0 + max(0, motion_sim)*0.5)
            
        if is_match and score > best_score:
            best_score, best_tid = score, old_tid
            
    return best_tid

def _inherit_id(new_tid, old_tid):
    oid = _locked[old_tid]
    team = _team_locked.get(old_tid, _team_snapshot.get(old_tid, 0))
    _locked[new_tid] = oid
    _team_locked[new_tid] = team
    _team_snapshot[new_tid] = team
    _emb_history[new_tid] = _emb_history.pop(old_tid, deque(maxlen=30))
    _vel_history[new_tid] = _vel_history.pop(old_tid, deque(maxlen=10))
    print(f"[REID] Track {new_tid} inherited {oid} from {old_tid}")

def try_assign_ids() -> Dict[int, str]:
    """Attempts to lock official IDs for stable tracks."""
    for tid, pwin in _presence.items():
        if tid in _locked: continue
        if len(pwin) < STABILITY_WINDOW or sum(pwin) < STABILITY_THRESHOLD: continue
        if tid not in _team_snapshot: continue
        
        team = _team_snapshot[tid]
        if _team_counters[team] > MAX_PER_TEAM: continue
        
        oid = f"T{team+1}#{_team_counters[team]}"
        _locked[tid] = oid
        _team_locked[tid] = team
        _team_counters[team] += 1
        print(f"[LOCKED] Track {tid} -> {oid}")
        
    return dict(_locked)

def cleanup_stale_ids():
    """Prunes transient track metadata to prevent memory leaks."""
    to_prune = [tid for tid, last_f in _last_frame.items() 
                if (_current_frame - last_f) > STALE_THRESHOLD and tid not in _locked]
    for tid in to_prune:
        for d in [_last_box, _presence, _team_snapshot, _last_frame, _vel_history, _emb_history]:
            if tid in d: del d[tid]
    if to_prune: print(f"[GC] Pruned {len(to_prune)} transient tracks.")

def get_official_id(tid: int) -> Optional[str]:
    return _locked.get(tid)

def reset():
    global _locked, _team_locked, _team_counters, _presence, _team_snapshot
    global _last_box, _emb_history, _vel_history, _last_frame, _current_frame
    _locked, _team_locked = {}, {}
    _team_counters = {0: 1, 1: 1}
    _presence = defaultdict(lambda: deque(maxlen=STABILITY_WINDOW))
    _team_snapshot, _last_box = {}, {}
    _emb_history = defaultdict(lambda: deque(maxlen=30))
    _vel_history = defaultdict(lambda: deque(maxlen=10))
    _last_frame, _current_frame = {}, 0
