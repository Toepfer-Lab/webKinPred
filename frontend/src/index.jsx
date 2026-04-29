// src/index.js

import React from 'react';
import ReactDOM from 'react-dom/client';
import 'bootstrap/dist/css/bootstrap.min.css';
import './styles/global.css';
import './styles/layout.css';
import './styles/components/button.css';
import './styles/components/form.css';
import './styles/components/invalid-list.css';
import './styles/components/JobSubmissionForm.css';
import './styles/light-mode.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

import { ensureCsrfCookie } from './components/appClient';

const UMAMI_SCRIPT_SRC = 'https://analytics.humanmetabolism.org/script.js';
const UMAMI_WEBSITE_ID = 'bc4ea3b4-c4d9-46d3-8823-63b978134cf3';

function injectAnalyticsScript() {
  if (import.meta.env.VITE_ENABLE_ANALYTICS === 'false') return;
  if (document.querySelector(`script[data-website-id="${UMAMI_WEBSITE_ID}"]`)) return;

  const script = document.createElement('script');
  script.async = true;
  script.src = UMAMI_SCRIPT_SRC;
  script.setAttribute('data-website-id', UMAMI_WEBSITE_ID);
  document.head.appendChild(script);
}

function loadAnalyticsOffCriticalPath() {
  const run = () => window.setTimeout(injectAnalyticsScript, 0);

  if ('requestIdleCallback' in window) {
    window.requestIdleCallback(run, { timeout: 4000 });
    return;
  }

  window.setTimeout(run, 2000);
}

ensureCsrfCookie(); // fire and forget

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

reportWebVitals();
loadAnalyticsOffCriticalPath();
