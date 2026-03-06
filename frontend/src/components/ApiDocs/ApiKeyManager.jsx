/**
 * ApiKeyManager — self-service API key widget shown on the /api-docs page.
 *
 * States:
 *   loading  — checking whether the visitor already has a key
 *   no-key   — visitor has no key → show "Generate" button
 *   new-key  — key was just generated → show full key + copy button
 *   has-key  — visitor already has a key → show masked suffix + revoke/regenerate
 */

import React, { useEffect, useState, useCallback } from 'react';

export default function ApiKeyManager() {
  const [state, setState] = useState('loading'); // loading | no-key | new-key | has-key
  const [keySuffix, setKeySuffix] = useState('');
  const [fullKey, setFullKey] = useState('');
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  // ── Check key status on mount ────────────────────────────────────────────
  const checkStatus = useCallback(async () => {
    try {
      const resp = await fetch('/api/api-key/');
      const data = await resp.json();
      if (data.hasKey) {
        setKeySuffix(data.keySuffix);
        setState('has-key');
      } else {
        setState('no-key');
      }
    } catch {
      setState('no-key');
    }
  }, []);

  useEffect(() => { checkStatus(); }, [checkStatus]);

  // ── Generate ─────────────────────────────────────────────────────────────
  const handleGenerate = async () => {
    setBusy(true);
    setError('');
    try {
      const resp = await fetch('/api/api-key/generate/', { method: 'POST' });
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.error || 'Failed to generate key.');
        setBusy(false);
        return;
      }
      setFullKey(data.key);
      setKeySuffix(data.keySuffix);
      setState('new-key');
    } catch {
      setError('Network error. Please try again.');
    }
    setBusy(false);
  };

  // ── Revoke ───────────────────────────────────────────────────────────────
  const handleRevoke = async () => {
    setBusy(true);
    setError('');
    try {
      const resp = await fetch('/api/api-key/revoke/', { method: 'POST' });
      if (!resp.ok) {
        const data = await resp.json();
        setError(data.error || 'Failed to revoke key.');
        setBusy(false);
        return;
      }
      setFullKey('');
      setKeySuffix('');
      setCopied(false);
      setState('no-key');
    } catch {
      setError('Network error. Please try again.');
    }
    setBusy(false);
  };

  // ── Copy ─────────────────────────────────────────────────────────────────
  const handleCopy = () => {
    navigator.clipboard.writeText(fullKey).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    });
  };

  // ── Render ───────────────────────────────────────────────────────────────
  if (state === 'loading') {
    return (
      <div className="api-key-box">
        <span className="api-key-label">Checking for existing API key…</span>
      </div>
    );
  }

  if (state === 'new-key') {
    return (
      <div className="api-key-box api-key-box-success">
        <span className="api-key-label">Your API key (shown only once):</span>
        <div className="api-key-display">
          <code className="api-key-value">{fullKey}</code>
          <button
            className={`api-key-btn api-key-copy-btn${copied ? ' copied' : ''}`}
            onClick={handleCopy}
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
        <p className="api-key-warning">
          Save this key now — you will not be able to see it again.
          If you lose it, you can revoke it and generate a new one.
        </p>
      </div>
    );
  }

  if (state === 'has-key') {
    return (
      <div className="api-key-box">
        <span className="api-key-label">
          API key: <code>ak_•••••{keySuffix}</code>
        </span>
        <div className="api-key-actions">
          <button
            className="api-key-btn api-key-revoke-btn"
            onClick={handleRevoke}
            disabled={busy}
          >
            {busy ? 'Revoking…' : 'Revoke & Generate New Key'}
          </button>
        </div>
        {error && <p className="api-key-error">{error}</p>}
      </div>
    );
  }

  // state === 'no-key'
  return (
    <div className="api-key-box">
      <span className="api-key-label">
        You don't have an API key yet. Generate one to get started.
      </span>
      <div className="api-key-actions">
        <button
          className="api-key-btn api-key-generate-btn"
          onClick={handleGenerate}
          disabled={busy}
        >
          {busy ? 'Generating…' : 'Generate API Key'}
        </button>
      </div>
      {error && <p className="api-key-error">{error}</p>}
    </div>
  );
}
