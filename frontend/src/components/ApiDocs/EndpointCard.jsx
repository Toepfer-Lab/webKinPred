/**
 * EndpointCard — collapsible documentation card for a single API endpoint.
 *
 * Props:
 *   method       {string}    — "GET" or "POST"
 *   path         {string}    — e.g. "/api/v1/submit/"
 *   summary      {string}    — One-line description shown in the collapsed header
 *   requiresAuth {boolean}   — Whether the endpoint needs a Bearer token
 *   children     {ReactNode} — Expanded content (description, params, examples)
 */

import React, { useState } from 'react';

export default function EndpointCard({ method, path, summary, requiresAuth = true, children }) {
  const [open, setOpen] = useState(false);
  const [pathCopied, setPathCopied] = useState(false);

  function handleCopyPath(e) {
    e.stopPropagation();
    navigator.clipboard.writeText(path).then(() => {
      setPathCopied(true);
      setTimeout(() => setPathCopied(false), 2000);
    });
  }

  return (
    <div className="endpoint-card">
      <div
        className="endpoint-header"
        onClick={() => setOpen(o => !o)}
        role="button"
        tabIndex={0}
        onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && setOpen(o => !o)}
        aria-expanded={open}
      >
        <span className={`endpoint-method method-${method.toLowerCase()}`}>
          {method}
        </span>
        <span className="endpoint-path">{path}</span>
        <button
          type="button"
          className={`endpoint-copy-path-btn${pathCopied ? ' copied' : ''}`}
          onClick={handleCopyPath}
          title="Copy path"
        >
          {pathCopied ? '✓' : '⧉'}
        </button>
        <span className="endpoint-summary">
          {summary}
        </span>
        {requiresAuth ? (
          <span className="endpoint-auth-badge">Auth required</span>
        ) : (
          <span className="endpoint-no-auth-badge">No auth</span>
        )}
        <span className={`endpoint-chevron${open ? ' open' : ''}`}>▼</span>
      </div>

      {open && (
        <div className="endpoint-body">
          {children}
        </div>
      )}
    </div>
  );
}
