/**
 * TabbedCode — displays the same example in multiple languages via tabs.
 *
 * Props:
 *   tabs  {Array<{ label: string, code: string }>}  — One entry per language tab.
 */

import React, { useState } from 'react';

export default function TabbedCode({ tabs }) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [copied, setCopied] = useState(false);

  const current = tabs[activeIndex];

  function handleCopy() {
    navigator.clipboard.writeText(current.code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className="code-tabs">
      <div className="code-tab-bar">
        {tabs.map((tab, i) => (
          <button
            key={tab.label}
            type="button"
            className={`code-tab-btn${i === activeIndex ? ' active' : ''}`}
            onClick={() => { setActiveIndex(i); setCopied(false); }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="code-tab-content">
        <button
          type="button"
          className={`code-tab-copy-btn${copied ? ' copied' : ''}`}
          onClick={handleCopy}
        >
          {copied ? '✓ Copied' : 'Copy'}
        </button>
        <pre className="code-tab-block">{current.code}</pre>
      </div>
    </div>
  );
}
