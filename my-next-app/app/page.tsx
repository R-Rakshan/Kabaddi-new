"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import styles from "./page.module.css";

const API = "http://localhost:8001/api";

// ── Types ──────────────────────────────────────────────────────────────────

interface Job {
  id: string;
  input_filename: string;
  status: "queued" | "processing" | "done" | "error";
  progress: number;
  total_frames: number;
  processed_frames: number;
  elapsed_seconds: number;
  created_at: string;
}

interface Player {
  id: number;
  job_id: string;
  track_id: number;
  official_id: string;
  team: number;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: Job["status"] }) {
  const map: Record<string, { label: string; cls: string }> = {
    queued: { label: "Queued", cls: styles.badgeQueued },
    processing: { label: "Processing", cls: styles.badgeProcessing },
    done: { label: "Done", cls: styles.badgeDone },
    error: { label: "Error", cls: styles.badgeError },
  };
  const { label, cls } = map[status] ?? { label: status, cls: "" };
  return <span className={`${styles.badge} ${cls}`}>{label}</span>;
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div className={styles.progressTrack}>
      <div className={styles.progressFill} style={{ width: `${value}%` }} />
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function Home() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Fetch jobs list ────────────────────────────────────────────────────
  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch(`${API}/jobs`);
      if (!res.ok) return;
      const data: Job[] = await res.json();
      setJobs(data);
      // Update selected job if present
      setSelectedJob((prev) => {
        if (!prev) return prev;
        return data.find((j) => j.id === prev.id) ?? prev;
      });
    } catch { }
  }, []);

  // ── Fetch players for selected job ─────────────────────────────────────
  const fetchPlayers = useCallback(async (jobId: string) => {
    try {
      const res = await fetch(`${API}/jobs/${jobId}/players`);
      if (!res.ok) return;
      const data = await res.json();
      setPlayers(data.players ?? []);
    } catch { }
  }, []);

  // ── Polling ────────────────────────────────────────────────────────────
  useEffect(() => {
    fetchJobs();
    pollRef.current = setInterval(fetchJobs, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [fetchJobs]);

  useEffect(() => {
    if (!selectedJob) return;
    fetchPlayers(selectedJob.id);
  }, [selectedJob, fetchPlayers]);

  // ── Upload ─────────────────────────────────────────────────────────────
  const handleUpload = async (file: File) => {
    if (!file) return;
    setError("");
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/jobs/upload`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json();
        setError(err.detail ?? "Upload failed");
        return;
      }
      await fetchJobs();
    } catch (e) {
      setError("Cannot reach backend. Is it running on port 8000?");
    } finally {
      setUploading(false);
    }
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleUpload(file);
  };

  // ── Delete job ─────────────────────────────────────────────────────────
  const deleteJob = async (jobId: string) => {
    await fetch(`${API}/jobs/${jobId}`, { method: "DELETE" });
    if (selectedJob?.id === jobId) { setSelectedJob(null); setPlayers([]); }
    fetchJobs();
  };

  // ── Team split for display ─────────────────────────────────────────────
  const team1 = players.filter((p) => p.team === 1).sort((a, b) => a.official_id.localeCompare(b.official_id));
  const team2 = players.filter((p) => p.team === 2).sort((a, b) => a.official_id.localeCompare(b.official_id));

  return (
    <div className={styles.root}>
      {/* ── Sidebar ───────────────────────────────────────── */}
      <aside className={styles.sidebar}>
        <div className={styles.logo}>
          <span className={styles.logoIcon}>⚡</span>
          <div>
            <div className={styles.logoTitle}>Kabaddi Vision</div>
            <div className={styles.logoSub}>AI Player Tracking</div>
          </div>
        </div>

        {/* Upload drop zone */}
        <div
          className={`${styles.dropzone} ${dragOver ? styles.dropzoneActive : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileRef.current?.click()}
        >
          <input ref={fileRef} type="file" accept=".mp4,.avi,.mov,.mkv" hidden onChange={onFileChange} />
          {uploading ? (
            <div className={styles.uploadSpinner} />
          ) : (
            <>
              <div className={styles.dropIcon}>🎬</div>
              <div className={styles.dropLabel}>Drop video or click to upload</div>
              <div className={styles.dropSub}>.mp4 / .avi / .mov / .mkv</div>
            </>
          )}
        </div>

        {error && <div className={styles.errorBanner}>{error}</div>}

        {/* Job list */}
        <div className={styles.sidebarSection}>
          <div className={styles.sidebarHeading}>Recent Jobs</div>
          {jobs.length === 0 ? (
            <div className={styles.empty}>No jobs yet</div>
          ) : (
            jobs.map((job) => (
              <div
                key={job.id}
                className={`${styles.jobCard} ${selectedJob?.id === job.id ? styles.jobCardActive : ""}`}
                onClick={() => setSelectedJob(job)}
              >
                <div className={styles.jobCardTop}>
                  <span className={styles.jobName}>{job.input_filename}</span>
                  <StatusBadge status={job.status} />
                </div>
                {job.status === "processing" && (
                  <ProgressBar value={job.progress} />
                )}
                <div className={styles.jobMeta}>
                  {job.status === "processing"
                    ? `${job.progress}% · ${job.processed_frames}/${job.total_frames} frames`
                    : job.status === "done"
                      ? `${job.processed_frames} frames · ${job.elapsed_seconds}s`
                      : new Date(job.created_at).toLocaleString()}
                </div>
              </div>
            ))
          )}
        </div>
      </aside>

      {/* ── Main content ──────────────────────────────────── */}
      <main className={styles.main}>
        {!selectedJob ? (
          <div className={styles.hero}>
            <div className={styles.heroGlow} />
            <div className={styles.heroIcon}>🏃</div>
            <h1 className={styles.heroTitle}>Kabaddi Player Tracking</h1>
            <p className={styles.heroSub}>
              Upload a match video to auto-detect and track all 14 players using
              YOLOv8 + ByteTrack, classify teams by jersey color, and export
              a fully annotated video with stable IDs.
            </p>
            <div className={styles.heroPills}>
              <span className={styles.pill}>YOLOv8 Detection</span>
              <span className={styles.pill}>ByteTrack</span>
              <span className={styles.pill}>Pose Estimation</span>
              <span className={styles.pill}>Team Classification</span>
              <span className={styles.pill}>SQLite</span>
            </div>
          </div>
        ) : (
          <div className={styles.detail}>
            {/* Header */}
            <div className={styles.detailHeader}>
              <div>
                <h2 className={styles.detailTitle}>{selectedJob.input_filename}</h2>
                <div className={styles.detailMeta}>
                  Job ID: <span className="mono">{selectedJob.id}</span>
                </div>
              </div>
              <div className={styles.detailActions}>
                <StatusBadge status={selectedJob.status} />
                {selectedJob.status === "done" && (
                  <a
                    href={`${API}/jobs/${selectedJob.id}/video`}
                    download
                    className={styles.btnPrimary}
                  >
                    ⬇ Download Video
                  </a>
                )}
                <button
                  className={styles.btnDanger}
                  onClick={() => deleteJob(selectedJob.id)}
                >
                  Delete
                </button>
              </div>
            </div>

            {/* Progress */}
            {selectedJob.status === "processing" && (
              <div className={styles.progressSection}>
                <div className={styles.progressLabel}>
                  Processing… {selectedJob.progress}%
                  &nbsp;·&nbsp;{selectedJob.processed_frames}/{selectedJob.total_frames} frames
                </div>
                <ProgressBar value={selectedJob.progress} />
              </div>
            )}

            {/* Stats row */}
            {selectedJob.status === "done" && (
              <div className={styles.statsRow}>
                <div className={styles.statCard}>
                  <div className={styles.statValue}>{selectedJob.processed_frames}</div>
                  <div className={styles.statLabel}>Frames Processed</div>
                </div>
                <div className={styles.statCard}>
                  <div className={styles.statValue}>{selectedJob.elapsed_seconds}s</div>
                  <div className={styles.statLabel}>Processing Time</div>
                </div>
                <div className={styles.statCard}>
                  <div className={styles.statValue}>{players.length}</div>
                  <div className={styles.statLabel}>Players Identified</div>
                </div>
                <div className={styles.statCard}>
                  <div className={styles.statValue}>
                    {selectedJob.total_frames > 0
                      ? (selectedJob.processed_frames / selectedJob.elapsed_seconds).toFixed(1)
                      : "—"}
                  </div>
                  <div className={styles.statLabel}>FPS Throughput</div>
                </div>
              </div>
            )}

            {/* Team rosters */}
            {selectedJob.status === "done" && (
              <div className={styles.teamsRow}>
                {/* Team 1 */}
                <div className={`${styles.teamPanel} ${styles.teamPanelRed}`}>
                  <div className={styles.teamHeader}>
                    <span className={`${styles.teamDot} ${styles.teamDotRed}`} />
                    Team 1
                  </div>
                  <div className={styles.rosterGrid}>
                    {team1.length === 0
                      ? <div className={styles.empty}>No players assigned yet</div>
                      : team1.map((p) => (
                        <div key={p.id} className={`${styles.playerChip} ${styles.playerChipRed}`}>
                          <span className={styles.chipId}>{p.official_id}</span>
                          <span className={styles.chipTrack}>Track #{p.track_id}</span>
                        </div>
                      ))}
                  </div>
                </div>

                {/* Team 2 */}
                <div className={`${styles.teamPanel} ${styles.teamPanelBlue}`}>
                  <div className={styles.teamHeader}>
                    <span className={`${styles.teamDot} ${styles.teamDotBlue}`} />
                    Team 2
                  </div>
                  <div className={styles.rosterGrid}>
                    {team2.length === 0
                      ? <div className={styles.empty}>No players assigned yet</div>
                      : team2.map((p) => (
                        <div key={p.id} className={`${styles.playerChip} ${styles.playerChipBlue}`}>
                          <span className={styles.chipId}>{p.official_id}</span>
                          <span className={styles.chipTrack}>Track #{p.track_id}</span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}

            {/* Error state */}
            {selectedJob.status === "error" && (
              <div className={styles.errorBox}>
                ⚠️ Pipeline encountered an error. Check backend logs.
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
