import React from 'react';
import PropTypes from 'prop-types';
import { Alert } from 'react-bootstrap';

/**
 * Display publication details for a prediction method.
 *
 * Props
 * -----
 * methodKey   : string  – registry key, e.g. "DLKcat"
 * methods     : object  – the full methods dict from the backend registry
 *                         (keyed by method key, each entry has displayName,
 *                          publicationTitle, citationUrl, moreInfo, …)
 * citationOnly: bool    – when true, hides moreInfo and only shows citation
 */
export default function MethodDetails({ methodKey, methods, citationOnly = false }) {
  if (!methods || !methodKey) return null;
  const method = methods[methodKey];
  if (!method) return null;

  return (
    <Alert variant="info" className="mt-3">
      {method.publicationTitle && method.citationUrl && (
        <p>
          <strong>Publication: </strong>
          <a href={method.citationUrl} target="_blank" rel="noopener noreferrer">
            {method.publicationTitle}
          </a>
        </p>
      )}
      {!citationOnly && method.moreInfo && (
        <p><strong>Note: </strong>{method.moreInfo}</p>
      )}
    </Alert>
  );
}

MethodDetails.propTypes = {
  methodKey: PropTypes.string,
  methods: PropTypes.object,
  citationOnly: PropTypes.bool,
};
