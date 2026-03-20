import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import '../styles/components/NotFound.css';

export default function NotFound() {
  const location = useLocation();
  const requestedPath = `${location.pathname}${location.search}${location.hash}`;

  return (
    <section className="not-found-wrap" aria-labelledby="not-found-title">
      <div className="not-found-card">
        <p className="not-found-code">404</p>
        <h1 id="not-found-title" className="not-found-title">Page not found</h1>
        <p className="not-found-copy">
          The page you requested is not available. It may have moved, or the URL may be incorrect.
        </p>
        <p className="not-found-path-label">Requested path</p>
        <code className="not-found-path">{requestedPath}</code>
        <div className="not-found-actions">
          <Link className="not-found-link not-found-link-primary" to="/">Go to Predictor</Link>
          <Link className="not-found-link" to="/track-job">Track a Job</Link>
        </div>
      </div>
    </section>
  );
}
