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
  Stopwatch
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
  const [timeElapsed, setTimeElapsed] = useState('');
  // server-provided elapsed seconds (integer). We'll increment locally while active.
  const [elapsedSeconds, setElapsedSeconds] = useState(null);
  const [isCopying, setIsCopying] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Persist progress so x/y never drops to 0/0 when the job completes
  const [metrics, setMetrics] = useState({
    moleculesProcessed: 0,
    totalMolecules: 0,
    predictionsMade: 0,
    totalPredictions: 0,
    invalidRows: 0,
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

        // accept server-calculated elapsed seconds if provided
        if (data.elapsed_seconds != null) {
          setElapsedSeconds(Number(data.elapsed_seconds));
        }

        if (!isMounted.current) return;

        setJobStatus(data);
        setError(null);

        setMetrics(() => ({
          moleculesProcessed: data.molecules_processed,
          totalMolecules: data.total_molecules,
          predictionsMade: data.predictions_made,
          totalPredictions: data.total_predictions,
          invalidRows: data.invalid_rows,
        }));

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
    // Reset sticky metrics for a new job
    setMetrics({
      moleculesProcessed: 0,
      totalMolecules: 0,
      predictionsMade: 0,
      totalPredictions: 0,
      invalidRows: 0,
    });
    if (publicId) fetchJobStatus(publicId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [publicId]);

  // Elapsed time updater
  useEffect(() => {
    if (!(jobStatus && jobStatus.submission_time)) return;

    // Use server-provided elapsedSeconds to render initial value. While job is active
    // increment locally for smooth UI updates between polls.
    if (elapsedSeconds == null) return;

    setTimeElapsed(formatDuration(moment.duration(elapsedSeconds * 1000)));

    const active = jobStatus && (jobStatus.status === 'Processing' || jobStatus.status === 'Pending');
    if (!active) return;

    const id = setInterval(() => {
      setElapsedSeconds((s) => {
        const next = (s || 0) + 1;
        setTimeElapsed(formatDuration(moment.duration(next * 1000)));
        return next;
      });
    }, 1000);

    return () => clearInterval(id);
  }, [elapsedSeconds, jobStatus]);

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

  // Percentages based on sticky metrics
  const moleculesPct = useMemo(() => {
    const done = metrics.moleculesProcessed || 0;
    const total = metrics.totalMolecules || 0;
    return total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  }, [metrics]);

  const predsPct = useMemo(() => {
    const made = metrics.predictionsMade || 0;
    const total = metrics.totalPredictions || 0;
    return total > 0 ? Math.min(100, Math.round((made / total) * 100)) : 0;
  }, [metrics]);

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
    <Container className="mt-5 pb-5">
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
                        placeholder="e.g., 8f4e7a9b-1234-4acb-9d01-abcdef123456"
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
                <div className="d-flex flex-wrap align-items-center justify-content-between gap-2 mb-3">
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
                  {/* Stats always show sticky x/y, even after completion */}
                  <Row className="mb-3 g-3 stats-grid">
                    <Col sm={6} lg={3}>
                      <div className="stat-card">
                        <div className="stat-label"><Stopwatch className="me-2" />Time Elapsed</div>
                        <div className="stat-value">{timeElapsed || '—'}</div>
                      </div>
                    </Col>
                    <Col sm={6} lg={3}>
                      <div className="stat-card">
                        <div className="stat-label">Preprocessed</div>
                        <div className="stat-value">
                          {metrics.moleculesProcessed}
                          <span className="stat-sub"> / {metrics.totalMolecules}</span>
                        </div>
                      </div>
                    </Col>
                    <Col sm={6} lg={3}>
                      <div className="stat-card">
                        <div className="stat-label">Predictions</div>
                        <div className="stat-value">
                          {metrics.predictionsMade}
                          <span className="stat-sub"> / {metrics.totalPredictions}</span>
                        </div>
                      </div>
                    </Col>
                    <Col sm={6} lg={3}>
                      <div className="stat-card">
                        <div className="stat-label">Invalid Rows</div>
                        <div className="stat-value">{metrics.invalidRows}</div>
                      </div>
                    </Col>
                  </Row>

                  {(jobStatus.status === 'Processing') && (
                    <>
                      {metrics.totalMolecules > 0 && (
                        <div className="mb-3">
                          <div className="progress-row">
                            <div className="progress-title">Reactions Processed</div>
                            <div className="progress-count">{moleculesPct}%</div>
                          </div>
                          <ProgressBar now={moleculesPct} className="kave-progress" />
                        </div>
                      )}

                      {metrics.totalPredictions > 0 && (
                        <div className="mb-2">
                          <div className="progress-row">
                            <div className="progress-title">Predictions Made</div>
                            <div className="progress-count">{predsPct}%</div>
                          </div>
                          <ProgressBar now={predsPct} className="kave-progress" />
                        </div>
                      )}
                      <div className="mt-3 d-flex justify-content-center">
                        <Spinner animation="border" role="status" />
                      </div>
                    </>
                  )}

                  {jobStatus.status === 'Pending' && (
                    <div className="mt-3 d-flex flex-column align-items-center">
                      <div className="stat-label mb-2">Job is queued and waiting to start...</div>
                      <Spinner animation="border" role="status" />
                    </div>
                  )}

                  {jobStatus.status === 'Completed' && (
                    <div className="d-flex align-items-center justify-content-between flex-wrap gap-2 mt-3">
                      <div>Job completed. Download your results below.</div>
                        <a
                          className="btn btn-custom-subtle"
                          href={`${apiBaseUrl}/jobs/${publicId}/download/`}
                        >
                          <FileEarmarkArrowDown className="me-2" />
                          Download Results
                        </a>
                    </div>
                  )}

                  {/* Expandable details for skipped/unpredicted rows — only on Completed */}
                  {skippedRowsMessage && (
                    <div className="mt-3">
                      <ExpandableErrorMessage errorMessage={skippedRowsMessage} />
                    </div>
                  )}

                  {/* User-friendly failure banner */}
                  {jobStatus.status === 'Failed' && (
                    <div className="job-failed-banner mt-3">
                      <div className="job-failed-header">
                        <XCircle size={20} className="me-2" />
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
