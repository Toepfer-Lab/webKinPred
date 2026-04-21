import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import {
  Form,
  Container,
  Row,
  Col,
  Card,
  Alert,
  ProgressBar,
  Spinner,
  Button,
  Badge
} from 'react-bootstrap';
import {
  HourglassSplit,
  CheckCircle,
  XCircle,
  ExclamationTriangle,
  Clipboard,
  ClipboardCheck,
  ArrowClockwise,
  FileEarmarkArrowDown,
  Stopwatch,
  Database,
  Cpu,
  GraphUp,
  CheckCircleFill,
} from 'react-bootstrap-icons';
import moment from 'moment';
import ExpandableErrorMessage from './ExpandableErrorMessage';
import apiClient from './appClient';
import '../styles/components/JobStatus.css';

function JobStatus() {
  const { public_id: routePublicId } = useParams();
  const [inputPublicId, setInputPublicId] = useState(routePublicId || '');
  const [publicId, setPublicId] = useState(routePublicId || '');

  const [jobStatus, setJobStatus] = useState(null);
  const [error, setError] = useState(null);
  const [queueTime, setQueueTime] = useState('');
  const [computeTime, setComputeTime] = useState('');
  const [queueSeconds, setQueueSeconds] = useState(null);
  const [computeSeconds, setComputeSeconds] = useState(null);
  const [queuePosition, setQueuePosition] = useState(null);
  const [isCopying, setIsCopying] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [progressStages, setProgressStages] = useState([]);

  // Persist progress so x/y never drops to 0/0 when the job completes
  const [metrics, setMetrics] = useState({
    moleculesProcessed: 0,
    totalMolecules: 0,
    predictionsMade: 0,
    totalPredictions: 0,
    invalidRows: 0,
    embeddingEnabled: false,
    embeddingState: '',
    embeddingMethodKey: '',
    embeddingTarget: '',
    embeddingTotal: 0,
    embeddingCachedAlready: 0,
    embeddingNeedComputation: 0,
    embeddingComputed: 0,
    embeddingRemaining: 0,
  });

  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api';

  // Polling control
  const timerRef = useRef(null);
  const isMounted = useRef(false);

  const clearTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  const scheduleNextPoll = useCallback(
    (delayMs) => {
      clearTimer();
      if (delayMs != null) {
        timerRef.current = setTimeout(() => {
          if (isMounted.current) fetchJobStatus(publicId);
        }, delayMs);
      }
    },
    [publicId]
  );

  const fetchJobStatus = useCallback(
    async (id, { manual = false } = {}) => {
      if (!id) return;
      if (manual) setIsRefreshing(true); // only for manual
      try {
        const response = await apiClient.get(`/job-status/${id}/`);
        const data = response.data;

        if (data.queue_seconds != null) {
          setQueueSeconds(Number(data.queue_seconds));
        }
        if (data.compute_seconds != null) {
          setComputeSeconds(Number(data.compute_seconds));
        }
        setQueuePosition(data.queue_position ?? null);

        if (!isMounted.current) return;

        setJobStatus(data);
        setError(null);

        setProgressStages(Array.isArray(data.progress_stages) ? data.progress_stages : []);

        setMetrics((prev) => {
          const embedding = data.embedding_progress;
          return {
            moleculesProcessed: data.molecules_processed,
            totalMolecules: data.total_molecules,
            predictionsMade: data.predictions_made,
            totalPredictions: data.total_predictions,
            invalidRows: data.invalid_rows,
            embeddingEnabled: embedding ? Boolean(embedding.enabled) : prev.embeddingEnabled,
            embeddingState: embedding ? (embedding.state || '') : prev.embeddingState,
            embeddingMethodKey: embedding
              ? (embedding.methodKey || embedding.method_key || '')
              : prev.embeddingMethodKey,
            embeddingTarget: embedding ? (embedding.target || '') : prev.embeddingTarget,
            embeddingTotal: embedding ? Number(embedding.total || 0) : prev.embeddingTotal,
            embeddingCachedAlready: embedding
              ? Number(embedding.cachedAlready ?? embedding.cached_already ?? 0)
              : prev.embeddingCachedAlready,
            embeddingNeedComputation: embedding
              ? Number(embedding.needComputation ?? embedding.need_computation ?? 0)
              : prev.embeddingNeedComputation,
            embeddingComputed: embedding ? Number(embedding.computed || 0) : prev.embeddingComputed,
            embeddingRemaining: embedding ? Number(embedding.remaining || 0) : prev.embeddingRemaining,
          };
        });

        const nextDelay =
          data.status === 'Processing' ? 1000 :
          data.status === 'Pending'    ? 3000 :
          null;

        scheduleNextPoll(nextDelay);
      } catch (err) {
        console.error('Error fetching job status:', err);
        if (isMounted.current) {
          setError('Unable to fetch job status. Retrying…');
          scheduleNextPoll(5000);
        }
      } finally {
        if (manual && isMounted.current) setIsRefreshing(false);
      }
    },
    [scheduleNextPoll]
  );


  // Mount / unmount
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      clearTimer();
    };
  }, []);

  // Kick off or restart polling only when the job ID changes
  useEffect(() => {
    clearTimer();
    setJobStatus(null);
    setError(null);
    setQueueSeconds(null);
    setComputeSeconds(null);
    setQueueTime('');
    setComputeTime('');
    setQueuePosition(null);
    setProgressStages([]);
    // Reset sticky metrics for a new job
    setMetrics({
      moleculesProcessed: 0,
      totalMolecules: 0,
      predictionsMade: 0,
      totalPredictions: 0,
      invalidRows: 0,
      embeddingEnabled: false,
      embeddingState: '',
      embeddingMethodKey: '',
      embeddingTarget: '',
      embeddingTotal: 0,
      embeddingCachedAlready: 0,
      embeddingNeedComputation: 0,
      embeddingComputed: 0,
      embeddingRemaining: 0,
    });
    if (publicId) fetchJobStatus(publicId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [publicId]);

  // Queue time ticker — increments while job is Pending
  useEffect(() => {
    if (queueSeconds == null) return;
    setQueueTime(formatDuration(moment.duration(queueSeconds * 1000)));
    if (jobStatus?.status !== 'Pending') return;
    const id = setInterval(() => {
      setQueueSeconds((s) => {
        const next = (s || 0) + 1;
        setQueueTime(formatDuration(moment.duration(next * 1000)));
        return next;
      });
    }, 1000);
    return () => clearInterval(id);
  }, [queueSeconds, jobStatus?.status]);

  // Compute time ticker — increments while job is Processing
  useEffect(() => {
    if (computeSeconds == null) return;
    setComputeTime(formatDuration(moment.duration(computeSeconds * 1000)));
    if (jobStatus?.status !== 'Processing') return;
    const id = setInterval(() => {
      setComputeSeconds((s) => {
        const next = (s || 0) + 1;
        setComputeTime(formatDuration(moment.duration(next * 1000)));
        return next;
      });
    }, 1000);
    return () => clearInterval(id);
  }, [computeSeconds, jobStatus?.status]);

  const handleCheckStatus = (e) => {
    e.preventDefault();
    if (!inputPublicId.trim()) return;
    setPublicId(inputPublicId.trim());
  };

  const handleManualRefresh = () => {
    if (publicId) {
      clearTimer();
      fetchJobStatus(publicId, { manual: true });
    }
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setIsCopying(true);
      setTimeout(() => setIsCopying(false), 1200);
    } catch {
      // ignore
    }
  };

  const statusMeta = useMemo(() => {
    const s = jobStatus?.status;
    if (s === 'Completed') return { variant: 'success', icon: <CheckCircle className="me-1" />, label: 'Completed' };
    if (s === 'Failed') return { variant: 'danger', icon: <XCircle className="me-1" />, label: 'Failed' };
    if (s === 'Processing') return { variant: 'info', icon: <HourglassSplit className="me-1" />, label: 'Processing' };
    if (s === 'Pending') return { variant: 'secondary', icon: <HourglassSplit className="me-1" />, label: 'Pending' };
    return { variant: 'secondary', icon: <HourglassSplit className="me-1" />, label: '—' };
  }, [jobStatus]);

  const normalizedStages = useMemo(() => {
    if (Array.isArray(progressStages) && progressStages.length > 0) {
      return progressStages;
    }
    if (!jobStatus) return [];
    if (jobStatus.status === 'Pending') return [];
    const stageStatus =
      jobStatus.status === 'Completed' ? 'completed' :
      jobStatus.status === 'Failed' ? 'failed' :
      jobStatus.status === 'Processing' ? 'running' :
      'pending';
    return [
      {
        index: 0,
        target: 'Prediction',
        method_name: '',
        method_key: '',
        status: stageStatus,
        prediction: {
          molecules_total: metrics.totalMolecules || 0,
          molecules_processed: metrics.moleculesProcessed || 0,
          invalid_rows: metrics.invalidRows || 0,
          predictions_total: metrics.totalPredictions || 0,
          predictions_made: metrics.predictionsMade || 0,
        },
        embedding: metrics.embeddingEnabled
          ? {
              enabled: true,
              state: metrics.embeddingState || 'running',
              total: metrics.embeddingTotal || 0,
              cached_already: metrics.embeddingCachedAlready || 0,
              need_computation: metrics.embeddingNeedComputation || 0,
              computed: metrics.embeddingComputed || 0,
              remaining: metrics.embeddingRemaining || 0,
            }
          : {
              enabled: false,
              state: 'not_required',
            },
      },
    ];
  }, [progressStages, jobStatus, metrics]);

  // Build a nice expandable block for rows we couldn't predict (if the API returns any flavour of this)
  const skippedRowsMessage = useMemo(() => {
    if (!jobStatus) return null;
    const raw = jobStatus.error_message;
    if (!raw) return null;

    // Try to parse backend's structured JSON: [{rows: [0,2], reason: "..."}, ...]
    if (typeof raw === 'string' && raw.trimStart().startsWith('[')) {
      try {
        const groups = JSON.parse(raw);
        if (Array.isArray(groups) && groups.length > 0) {
          const lines = groups.map(({ rows, reason }) => {
            const label = Array.isArray(rows) && rows.length > 0
              ? (rows.length === 1
                  ? `Row ${rows[0] + 1}`
                  : `Rows ${rows.map(r => r + 1).join(', ')}`)
              : 'Some rows';
            return `${label}: ${reason || 'Unknown reason'}`;
          });
          return lines.join('\n');
        }
      } catch {
        // fall through to plain string
      }
    }

    return typeof raw === 'string' ? raw : null;
  }, [jobStatus]);

  const failedMessage = useMemo(() => {
    if (jobStatus?.status !== 'Failed') return null;

    const candidates = [
      jobStatus?.error_message,
      jobStatus?.error,
      jobStatus?.message,
      jobStatus?.detail,
      jobStatus?.failure_reason,
    ];

    for (const value of candidates) {
      if (typeof value === 'string' && value.trim()) {
        return sanitiseErrorForUser(value.trim());
      }
      if (Array.isArray(value) && value.length > 0) {
        return sanitiseErrorForUser(value.map(String).join('\n'));
      }
    }

    return sanitiseErrorForUser('');
  }, [jobStatus]);

  return (
    <Container className="mt-1 pb-5">
      <Row className="justify-content-center">
        <Col md={10} lg={9}>
          <Card className="section-container job-status-card mb-4">
            <Card.Header as="h3" className="text-center">Track Job Status</Card.Header>

            <Card.Body>
              {!routePublicId && (
                <Form onSubmit={handleCheckStatus} className="mb-4">
                  <Form.Group controlId="jobIdInput">
                    <Form.Label className="mb-2">Enter Job ID</Form.Label>
                    <div className="d-flex gap-2">
                      <Form.Control
                        type="text"
                        value={inputPublicId}
                        placeholder="e.g., pl1a2V1"
                        onChange={(e) => setInputPublicId(e.target.value)}
                        required
                        className="kave-input"
                      />
                      <Button type="submit" className="kave-btn">
                        <span className="kave-line"></span>
                        Check Status
                      </Button>
                    </div>
                  </Form.Group>
                </Form>
              )}

              {error && (
                <Alert variant="warning" className="d-flex align-items-center">
                  <ExclamationTriangle className="me-2" />
                  <div>{error}</div>
                </Alert>
              )}

              {publicId && (
                <div className="job-header mb-4">
                  <div className="d-flex align-items-center gap-2">
                    <span className="label-muted">Job ID:</span>
                    <code className="jobid-chip">{publicId}</code>
                    <Button
                      variant="outline-light"
                      size="sm"
                      className="chip-action"
                      onClick={() => copyToClipboard(publicId)}
                    >
                      {isCopying ? <ClipboardCheck size={16} /> : <Clipboard size={16} />}
                    </Button>
                  </div>

                  <div className="d-flex align-items-center gap-2">
                    <Badge bg={statusMeta.variant} className="status-pill">
                      {statusMeta.icon}{statusMeta.label}
                    </Badge>
                    <Button
                      size="sm"
                      className="btn btn-custom-subtle"
                      onClick={handleManualRefresh}
                      disabled={isRefreshing}
                    >
                      <ArrowClockwise className={`me-1 ${isRefreshing ? 'spin' : ''}`} />
                      Refresh
                    </Button>
                  </div>
                </div>
              )}

              {jobStatus && (
                <>
                  {/* ── PENDING ────────────────────────────────────────── */}
                  {jobStatus.status === 'Pending' && (
                    <div className="pending-section mb-4">
                      {queuePosition != null && (
                        <div className="queue-position-badge">
                          Position <span className="queue-position-number">#{queuePosition}</span> in queue
                        </div>
                      )}
                      <div className="pending-hint">Job is queued and waiting to start…</div>
                      <Spinner animation="border" role="status" />
                    </div>
                  )}

                  {/* ── TIMING ─────────────────────────────────────────── */}
                  <div className="stat-section">
                    <div className="stat-section-title">Timing</div>
                    <Row className="g-3">
                      <Col xs={6}>
                        <div className="stat-card">
                          <div className="stat-label"><Stopwatch className="me-2" />Queue Time</div>
                          <div className="stat-value">{queueTime || '—'}</div>
                          <div className="stat-hint">Time spent waiting in queue before processing began</div>
                        </div>
                      </Col>
                      <Col xs={6}>
                        <div className="stat-card">
                          <div className="stat-label"><Cpu className="me-2" />Compute Time</div>
                          <div className="stat-value">{computeTime || '—'}</div>
                          <div className="stat-hint">Active processing time on the server</div>
                        </div>
                      </Col>
                    </Row>
                  </div>

                  {/* ── RESULTS ────────────────────────────────────────── */}
                  <div className="stat-section">
                    <div className="stat-section-title">Results</div>
                    {normalizedStages.length === 0 && (
                      <div className="stat-hint">
                        Target progress will appear once processing begins.
                      </div>
                    )}
                    {normalizedStages.map((stage) => {
                      const pred = stage.prediction || {};
                      const made = Number(pred.predictions_made ?? pred.predictionsMade ?? 0);
                      const total = Number(pred.predictions_total ?? pred.predictionsTotal ?? 0);
                      const processed = Number(pred.molecules_processed ?? pred.moleculesProcessed ?? 0);
                      const moleculesTotal = Number(pred.molecules_total ?? pred.moleculesTotal ?? 0);
                      const invalid = Number(pred.invalid_rows ?? pred.invalidRows ?? 0);
                      const pct = total > 0 ? Math.min(100, Math.round((made / total) * 100)) : 0;
                      const stageStatus = String(stage.status || 'pending').toLowerCase();
                      const badgeVariant =
                        stageStatus === 'completed' ? 'success'
                        : stageStatus === 'running' ? 'info'
                        : stageStatus === 'failed' ? 'danger'
                        : 'secondary';

                      return (
                        <div className="progress-item" key={`${stage.index}-${stage.target}-${stage.method_key || ''}`}>
                          <div className="progress-row">
                            <div className="progress-title">
                              <GraphUp className="me-2" />
                              {stage.target}
                              {(stage.method_name || stage.methodName) ? ` - ${stage.method_name || stage.methodName}` : ''}
                            </div>
                            <div className="progress-count">
                              <Badge bg={badgeVariant}>{stageStatus}</Badge>
                            </div>
                          </div>
                          <div className="progress-row">
                            <div className="progress-count">
                              Predictions: {made} / {total}
                            </div>
                            <div className="progress-count">
                              Validated: {processed} / {moleculesTotal}
                            </div>
                          </div>
                          <ProgressBar now={pct} className="kave-progress" />
                          <div className="stat-hint">
                            Invalid Rows: {invalid}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* ── PROTEIN EMBEDDINGS ─────────────────────────────── */}
                  <div className="stat-section">
                    <div className="stat-section-title">
                      <Database className="me-2" />Protein Embeddings
                    </div>
                    <p className="stat-section-desc">
                      Embedding progress is shown per target and method. Completed stages stay visible while the next target starts.
                    </p>
                    {normalizedStages.length === 0 && (
                      <div className="stat-hint">
                        Embedding details will appear once the active target starts prediction.
                      </div>
                    )}
                    {normalizedStages.map((stage) => {
                      const emb = stage.embedding || {};
                      const state = String(emb.state || (emb.enabled ? 'pending' : 'not_required')).toLowerCase();
                      const enabled = Boolean(emb.enabled);
                      const cached = Number(emb.cached_already ?? emb.cachedAlready ?? 0);
                      const need = Number(emb.need_computation ?? emb.needComputation ?? 0);
                      const computed = Number(emb.computed ?? 0);
                      const pct = need > 0 ? Math.min(100, Math.round((computed / need) * 100)) : 0;
                      const badgeVariant =
                        state === 'done' ? 'success'
                        : state === 'running' ? 'info'
                        : state === 'error' ? 'danger'
                        : state === 'not_required' ? 'secondary'
                        : 'secondary';

                      return (
                        <div className="progress-item" key={`emb-${stage.index}-${stage.target}-${stage.method_key || ''}`}>
                          <div className="progress-row">
                            <div className="progress-title">
                              <CheckCircleFill className="me-2" />
                              {stage.target}
                              {(stage.method_name || stage.methodName) ? ` - ${stage.method_name || stage.methodName}` : ''}
                            </div>
                            <div className="progress-count">
                              <Badge bg={badgeVariant}>{state}</Badge>
                            </div>
                          </div>
                          {!enabled || state === 'not_required' ? (
                            <div className="stat-hint">No embedding computation is required for this stage.</div>
                          ) : (
                            <>
                              <div className="progress-row">
                                <div className="progress-count">Cached: {cached}</div>
                                <div className="progress-count">
                                  Computed: {computed} / {need}
                                </div>
                              </div>
                              <ProgressBar now={pct} className="kave-progress" />
                            </>
                          )}
                        </div>
                      );
                    })}
                  </div>

                  {/* ── COMPLETED ──────────────────────────────────────── */}
                  {jobStatus.status === 'Completed' && (
                    <div className="job-complete-banner mt-3">
                      <div className="job-complete-header">
                        <CheckCircle size={18} className="me-2" />
                        Job Completed
                      </div>
                      <div className="job-complete-body d-flex align-items-center justify-content-between flex-wrap gap-2">
                        <span>Your results are ready for download.</span>
                        <a
                          className="btn btn-custom-subtle"
                          href={`${apiBaseUrl}/jobs/${publicId}/download/`}
                        >
                          <FileEarmarkArrowDown className="me-2" />
                          Download Results
                        </a>
                      </div>
                    </div>
                  )}

                  {/* Expandable details for skipped/unpredicted rows */}
                  {skippedRowsMessage && (
                    <div className="mt-3">
                      <ExpandableErrorMessage errorMessage={skippedRowsMessage} />
                    </div>
                  )}

                  {/* ── FAILED ─────────────────────────────────────────── */}
                  {jobStatus.status === 'Failed' && (
                    <div className="job-failed-banner mt-3">
                      <div className="job-failed-header">
                        <XCircle size={18} className="me-2" />
                        Job Failed
                      </div>
                      <div className="job-failed-body">
                        {failedMessage}
                      </div>
                    </div>
                  )}
                </>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default JobStatus;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Convert raw backend error strings into user-friendly messages.
 * Catches common patterns (OOM, subprocess failures, timeouts, missing data)
 * and replaces them with actionable, non-technical text.
 */
function sanitiseErrorForUser(raw) {
  if (!raw || typeof raw !== 'string') {
    return 'An unexpected error occurred while processing your job. Please try again or contact support.';
  }

  const lower = raw.toLowerCase();

  // Memory / OOM
  if (
    lower.includes('out of memory') ||
    lower.includes('memory') ||
    lower.includes('oom') ||
    lower.includes('sigkill') ||
    lower.includes('killed') ||
    lower.includes('ram') ||
    lower.includes('returncode == 137') ||
    lower.includes('exit status 137') ||
    lower.includes('returncode == -9') ||
    lower.includes('exit status -9')
  ) {
    return (
      'The prediction model ran out of memory while processing your data. ' +
      'This usually happens with very large or numerous protein sequences. ' +
      'Try reducing the number of rows or the length of your sequences and resubmit.'
    );
  }

  // Subprocess / non-zero exit (the exact issue the user reported)
  if (
    lower.includes('returned non-zero exit status') ||
    lower.includes('calledprocesserror') ||
    lower.includes('non-zero exit')
  ) {
    return (
      'The prediction model encountered an internal error and could not complete. ' +
      'This may be caused by unusually long sequences, unsupported characters in your input, ' +
      'or a temporary resource issue. Please verify your input data and try again.'
    );
  }

  // Timeout
  if (lower.includes('timeout') || lower.includes('timed out')) {
    return (
      'The prediction timed out. Your input may be too large for the selected model. ' +
      'Try reducing the number of rows and resubmitting.'
    );
  }

  // Missing columns
  if (lower.includes('missing column')) {
    return (
      'Your input file is missing one or more required columns. ' +
      'Please check the expected CSV format and resubmit.'
    );
  }

  // Failed to read CSV
  if (lower.includes('failed to read input csv')) {
    return (
      'The uploaded CSV file could not be read. ' +
      'Please ensure it is a valid CSV file and try again.'
    );
  }

  // If the message already looks clean (no paths, exit codes, tracebacks),
  // return it as-is. Otherwise fall back to a generic message.
  const hasInternalDetails =
    /\/[a-z_/]+\.[a-z]+/i.test(raw) ||  // file paths
    /exit status/i.test(raw) ||           // exit codes
    /traceback/i.test(raw) ||             // Python tracebacks
    /\bFile "/.test(raw);                 // Python stack frames

  if (hasInternalDetails) {
    return (
      'The prediction model encountered an unexpected error. ' +
      'Please verify your input data and try again. If the problem persists, contact support.'
    );
  }

  // Message looks user-safe — pass it through
  return raw;
}

function num(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
}
function formatDuration(duration) {
  if (!duration) return '';
  const hours = Math.floor(duration.asHours());
  const minutes = duration.minutes();
  const seconds = duration.seconds();
  return `${hours}h ${pad(minutes)}m ${pad(seconds)}s`;
}
function pad(n) {
  return String(n).padStart(2, '0');
}
