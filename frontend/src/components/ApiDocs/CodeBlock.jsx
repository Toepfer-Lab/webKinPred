/**
 * CodeBlock — a syntax-highlighted code block with a copy-to-clipboard button.
 *
 * Props:
 *   code     {string}  — The code to display.
 *   language {string}  — Label shown in the header bar (e.g. "bash", "python").
 */

import React, { useState } from 'react';

export default function CodeBlock({ code, language = '' }) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span>{language}</span>
        <button
          className={`code-block-copy-btn${copied ? ' copied' : ''}`}
          onClick={handleCopy}
          type="button"
        >
          {copied ? '✓ Copied' : 'Copy'}
        </button>
      </div>
      <pre className="code-block">{code}</pre>
    </div>
  );
}
