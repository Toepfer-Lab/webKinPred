// /home/saleh/webKinPred/frontend/src/components/job-submission/hooks/useJobSubmission.js
import { useState, useMemo, useRef, useEffect } from 'react';
import {
  detectCsvFormat,
  fetchMethods,
  validateCsv,
  fetchSequenceSimilaritySummary,
  submitJob as submitJobApi,
  openProgressStream,
  cancelValidationApi
} from '../services/api';

function makeSessionId() {
  return 'vs_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function useJobSubmission() {
  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [showPreprocessPrompt, setShowPreprocessPrompt] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [showValidationResults, setShowValidationResults] = useState(false);

  // Domain state
  const [predictionType, setPredictionType] = useState('');
  const [kcatMethod, setKcatMethod] = useState('');
  const [kmMethod, setKmMethod] = useState('');
  const [csvFile, setCsvFile] = useState(null);
  const [fileName, setFileName] = useState('No file chosen');
  const [useExperimental, setUseExperimental] = useState(false);
  const [handleLongSeqs, setHandleLongSeqs] = useState('truncate');

  // Derived server feedback
  const [csvFormatInfo, setCsvFormatInfo] = useState(null);
  const [csvFormatValid, setCsvFormatValid] = useState(false);
  const [csvFormatError, setCsvFormatError] = useState('');
  const [csvParsing, setCsvParsing] = useState(false);
  const [similarityData, setSimilarityData] = useState(null);
  const [submissionResult, setSubmissionResult] = useState(null);

  // Live log state
  const [validationSessionId, setValidationSessionId] = useState('');
  const userCancelledRef = useRef(false);
  const [liveLogs, setLiveLogs] = useState([]);
  const [streamConnected, setStreamConnected] = useState(false);
  const [showLogOverlay, setShowLogOverlay] = useState(false);
  const [validationDone, setValidationDone] = useState(false);
  const eventSourceRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // ── Method registry (fetched once from the backend) ────────────────────────
  // `methods` is a plain object keyed by method key, e.g.:
  //   { DLKcat: { supports: ["kcat"], inputFormat: "single", ... }, ... }
  const [methods, setMethods] = useState(null);

  useEffect(() => {
    fetchMethods()
      .then(setMethods)
      .catch((err) => {
        console.error('Failed to fetch method registry from backend:', err);
      });
  }, []);

  // ── Allowed methods derived from CSV format and registry ──────────────────

  /** Methods that support kcat and match the detected CSV input format. */
  const allowedKcatMethods = useMemo(() => {
    if (!csvFormatInfo?.csv_type || !methods) return [];
    return Object.entries(methods)
      .filter(([, m]) => m.supports.includes('kcat') && m.inputFormat === csvFormatInfo.csv_type)
      .map(([key]) => key);
  }, [csvFormatInfo, methods]);

  /** Methods that support Km (always single-substrate). */
  const allowedKmMethods = useMemo(() => {
    if (!methods) return [];
    return Object.entries(methods)
      .filter(([, m]) => m.supports.includes('Km'))
      .map(([key]) => key);
  }, [methods]);

  // ── Event handlers ────────────────────────────────────────────────────────

  const resetMethods = () => {
    setKcatMethod('');
    setKmMethod('');
  };

  const onChangePredictionType = (val) => {
    setPredictionType(val);
    resetMethods();
    setSimilarityData(null);
    setSubmissionResult(null);
    setShowValidationResults(false);
  };

  const onFileSelected = async (file) => {
    setCsvFile(file);
    setFileName(file?.name || 'No file chosen');
    resetMethods();
    setSimilarityData(null);
    setSubmissionResult(null);
    setShowValidationResults(false);

    if (!file) {
      setCsvFormatInfo(null);
      setCsvFormatValid(false);
      setCsvFormatError('');
      return;
    }

    setCsvParsing(true);
    try {
      const data = await detectCsvFormat(file);
      if (data.status === 'valid') {
        setCsvFormatInfo(data);
        setCsvFormatValid(true);
        setCsvFormatError('');
      } else {
        setCsvFormatInfo(null);
        setCsvFormatValid(false);
        setCsvFormatError(
          Array.isArray(data.errors) ? data.errors.join('; ') : 'Invalid CSV format.'
        );
      }
    } catch (err) {
      setCsvFormatInfo(null);
      setCsvFormatValid(false);
      setCsvFormatError(err?.response?.data?.error || 'Error detecting CSV format.');
    } finally {
      setCsvParsing(false);
    }
  };

  const openStream = (sid) => {
    if (eventSourceRef.current) {
      try { eventSourceRef.current.close(); } catch { /* ignore */ }
    }
    const es = openProgressStream(sid);
    eventSourceRef.current = es;
    setLiveLogs([]);
    setStreamConnected(true);

    es.onmessage = (evt) => {
      if (!evt?.data) return;
      setLiveLogs((prev) => [...prev, evt.data]);
    };
    es.addEventListener('done', () => {
      // Server signals the session is complete — close immediately so the
      // browser does not auto-reconnect and replay the log list again.
      es.close();
      eventSourceRef.current = null;
      setStreamConnected(false);
    });
    es.onerror = () => {
      setStreamConnected(false);
    };
  };

  const closeStream = () => {
    if (eventSourceRef.current) {
      try { eventSourceRef.current.close(); } catch { /* ignore */ }
      eventSourceRef.current = null;
    }
    setStreamConnected(false);
  };

  useEffect(() => {
    return () => closeStream();
  }, []);

  const closeLogOverlay = () => {
    setShowLogOverlay(false);
    setValidationDone(false);
    closeStream();
  };

  const runValidation = async () => {
    if (!csvFile) return;
    const sid = makeSessionId();
    setValidationSessionId(sid);
    userCancelledRef.current = false;
    setValidationDone(false);
    setShowLogOverlay(true);
    openStream(sid);
    setIsValidating(true);

    try {
      const validation = await validateCsv({
        file: csvFile,
        predictionType,
        kcatMethod,
        kmMethod,
      });
      const { invalid_substrates, invalid_proteins, length_violations } = validation;
      const simPromise = fetchSequenceSimilaritySummary({
        file: csvFile,
        useExperimental,
        validationSessionId: sid,
      });
      const sim = await simPromise;
      if (userCancelledRef.current) return;
      setSimilarityData(sim);
      setSubmissionResult({ invalid_substrates, invalid_proteins, length_violations });
      setShowValidationResults(true);
    } catch (err) {
      if (!userCancelledRef.current) {
        alert('Validation failed. Please try again. ' + (err?.message || ''));
      }
    } finally {
      setIsValidating(false);
      setValidationDone(false);
      setShowLogOverlay(false);
      setTimeout(() => closeStream(), 5000);
    }
  };

  const cancelValidation = async () => {
    if (!validationSessionId) {
      setIsValidating(false);
      setShowLogOverlay(false);
      closeStream();
      return;
    }
    userCancelledRef.current = true;
    try {
      await cancelValidationApi(validationSessionId);
    } catch { /* swallow */ }
    setIsValidating(false);
    setValidationDone(false);
    setShowLogOverlay(false);
    closeStream();
  };

  const submitJob = async () => {
    if (!csvFile) return;
    setIsSubmitting(true);
    try {
      const data = await submitJobApi({
        predictionType,
        kcatMethod,
        kmMethod,
        file: csvFile,
        handleLongSequences: handleLongSeqs,
        useExperimental,
      });
      setSubmissionResult((prev) => ({
        ...prev,
        message: data.message,
        public_id: data.public_id,
      }));
      setShowModal(true);
    } catch (error) {
      const msg = error?.response?.data?.error || '';
      alert('Failed to submit job\n' + msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return {
    // state
    isSubmitting,
    isValidating,
    showPreprocessPrompt,
    setShowPreprocessPrompt,
    showModal,
    setShowModal,
    showValidationResults,
    setShowValidationResults,

    predictionType,
    setPredictionType: onChangePredictionType,
    kcatMethod,
    setKcatMethod,
    kmMethod,
    setKmMethod,
    csvFile,
    fileName,
    csvFormatInfo,
    csvFormatValid,
    csvFormatError,
    csvParsing,
    useExperimental,
    setUseExperimental,
    handleLongSeqs,
    setHandleLongSeqs,
    similarityData,
    submissionResult,

    // registry-derived method lists
    methods,
    allowedKcatMethods,
    allowedKmMethods,

    // logs
    liveLogs,
    streamConnected,
    showLogOverlay,
    validationDone,
    autoScroll,
    setAutoScroll,

    // actions
    onFileSelected,
    runValidation,
    submitJob,
    cancelValidation,
    closeLogOverlay,
  };
}
