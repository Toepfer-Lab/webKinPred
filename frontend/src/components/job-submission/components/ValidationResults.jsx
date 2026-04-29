import React from 'react';
import PropTypes from 'prop-types';
import { Card, Button, Alert, Table, Form, Tabs, Tab } from 'react-bootstrap';
import {
  ChevronUp,
  ChevronDown,
  CheckCircleFill,
  XCircleFill,
  Rulers,
  BarChartFill,
} from 'react-bootstrap-icons';
import SequenceSimilarityHistogram from '../../SequenceSimilarityHistogram';
import InvalidItems from './InvalidItems';
import '../../../styles/components/ValidationResults.css'; // The new CSS file for tabs

export default function ValidationResults({
  submissionResult,
  showValidationResults,
  setShowValidationResults,
  handleLongSeqs,
  setHandleLongSeqs,
  similarityData,
  methods,
}) {
  const lengthViol = submissionResult?.length_violations || {};
  const lengthLimits = submissionResult?.length_limits || {};
  const toCount = (value) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
  };

  const serverViolations = toCount(lengthViol.Server);
  const serverLimit = Number(lengthLimits.Server) || 10000;

  const modelViolationRows = Object.entries(lengthViol)
    .filter(([key]) => key !== 'Server')
    .map(([key, value]) => {
      const methodMeta = methods?.[key];
      const hasLimitFromValidation = Object.prototype.hasOwnProperty.call(lengthLimits, key);
      const rawLimit = hasLimitFromValidation ? lengthLimits[key] : methodMeta?.maxSeqLen;
      const numericLimit = Number(rawLimit);
      const hasFiniteLimit = rawLimit !== null && Number.isFinite(numericLimit);

      return {
        key,
        label: methodMeta?.displayName || key,
        limitLabel:
          rawLimit === null
            ? '∞'
            : (hasFiniteLimit ? numericLimit.toLocaleString() : 'Unknown'),
        sortLimit: hasFiniteLimit ? numericLimit : Number.POSITIVE_INFINITY,
        violations: toCount(value),
      };
    })
    .filter((row) => row.violations > 0)
    .sort((a, b) => a.sortLimit - b.sortLimit || a.label.localeCompare(b.label));

  const hasAnyLengthIssues = modelViolationRows.length > 0 || serverViolations > 0;
  const hasInvalidItems =
    submissionResult?.invalid_substrates?.length > 0 || submissionResult?.invalid_proteins?.length > 0;

  return (
    <Card className="section-container section-validation-results mt-4">
      <Card.Header
        as="h3"
        className="text-center d-flex justify-content-center align-items-center position-relative"
        onClick={() => setShowValidationResults(!showValidationResults)}
        style={{ cursor: 'pointer' }}
      >
        Validation Summary
        <Button variant="outline-light" className="position-absolute end-0 me-3 border-0">
          {showValidationResults ? <ChevronUp /> : <ChevronDown />}
        </Button>
      </Card.Header>

      {showValidationResults && (
        <Card.Body>
          <Tabs defaultActiveKey={similarityData ? "similarity" : "validation"} id="validation-tabs" className="validation-tabs mb-4" justify>
            {/* Tab 1: Input Validation */}
            <Tab
              eventKey="validation"
              title={
                // MODIFIED: Added justify-content-center to center the icon and text
                <span className="d-flex align-items-center justify-content-center">
                  {hasInvalidItems ? (
                    <XCircleFill className="me-2 text-warning" />
                  ) : (
                    <CheckCircleFill className="me-2 text-success" />
                  )}
                  Input Validation
                </span>
              }
            >
              <div className="tab-content-wrapper">
                {hasInvalidItems ? (
                  <>
                    <Alert variant="warning">
                      ⛔ Some entries are invalid and will be skipped. You do not need to remove them from your file.
                    </Alert>
                    {submissionResult?.invalid_substrates?.length > 0 && (
                      <InvalidItems title="Invalid Substrates" items={submissionResult.invalid_substrates} />
                    )}
                    {submissionResult?.invalid_proteins?.length > 0 && (
                      <InvalidItems title="Invalid Proteins" items={submissionResult.invalid_proteins} />
                    )}
                  </>
                ) : (
                  <Alert variant="success">
                    ✅ All entries are valid. No issues were found with your input data.
                  </Alert>
                )}
              </div>
            </Tab>

            {/* Tab 2: Length Warnings (Conditional) */}
            {hasAnyLengthIssues && (
              <Tab
                eventKey="length"
                title={
                  // MODIFIED: Added justify-content-center
                  <span className="d-flex align-items-center justify-content-center">
                    <Rulers className="me-2 text-warning" />
                    Length Warnings
                  </span>
                }
              >
                <div className="tab-content-wrapper">
                  <h5 className="tab-section-header text-center mb-3">Protein Sequence Length Limits</h5>
                  <Table striped bordered hover size="sm" className="bg-dark">
                    <thead>
                      <tr>
                        <th className="text-white" colSpan="3" style={{ backgroundColor: '#4e4e4e' }}>
                          <strong>Model Limits</strong>
                        </th>
                      </tr>
                      <tr>
                        <th className="text-white">Category</th>
                        <th className="text-white">Limit</th>
                        <th className="text-white">Violations</th>
                      </tr>
                    </thead>
                    <tbody className="text-secondary">
                      {modelViolationRows.map(({ key, label, limitLabel, violations }) => (
                        <tr key={key}>
                          <td className="text-white">{label}</td>
                          <td className="text-white">{limitLabel}</td>
                          <td className="text-danger">{violations}</td>
                        </tr>
                      ))}
                    </tbody>
                    <tfoot>
                      {serverViolations > 0 && (
                        <tr style={{ borderTop: '2px solid rgb(78, 78, 78)' }}>
                          <td className="text-white">
                            <strong>Overall Server Limit</strong>
                          </td>
                          <td className="text-white">{serverLimit.toLocaleString()}</td>
                          <td className="text-danger">{serverViolations}</td>
                        </tr>
                      )}
                    </tfoot>
                  </Table>

                  <div className="mt-4 p-3 bg-light bg-opacity-10 rounded">
                    <p className="text-warning mb-3" style={{ fontSize: '1.05rem' }}>
                      <strong>How to handle long sequences?</strong>
                    </p>
                    <Form.Group>
                      <Form.Check
                        type="radio"
                        id="truncate-option"
                        name="longSeqHandling"
                        value="truncate"
                        label="Truncate sequences (default)"
                        checked={handleLongSeqs === 'truncate'}
                        onChange={() => setHandleLongSeqs('truncate')}
                      />
                      <Form.Text className="text-white-50 ms-4">
                        Truncation preserves the first and last portions of a sequence (e.g., for a 1000-limit, a 1200-length sequence becomes first 500 + last 500).
                      </Form.Text>
                      <div className="mt-3">
                        <Form.Check
                          type="radio"
                          id="skip-option"
                          name="longSeqHandling"
                          value="skip"
                          label="Skip sequences"
                          checked={handleLongSeqs === 'skip'}
                          onChange={() => setHandleLongSeqs('skip')}
                        />
                        <Form.Text className="text-white-50 ms-4">
                          Excludes any datapoint that contains a sequence exceeding length limits.
                        </Form.Text>
                      </div>
                    </Form.Group>
                  </div>
                </div>
              </Tab>
            )}

            {/* Tab 3: Similarity Analysis (Conditional) */}
            {similarityData && (
              <Tab
                eventKey="similarity"
                title={
                  <span className="d-flex align-items-center justify-content-center">
                    <BarChartFill className="me-2" />
                    Similarity Analysis
                  </span>
                }
              >
                <div className="tab-content-wrapper">
                  <SequenceSimilarityHistogram similarityData={similarityData} />
                </div>
              </Tab>
            )}
          </Tabs>
        </Card.Body>
      )}
    </Card>
  );
}

ValidationResults.propTypes = {
  submissionResult: PropTypes.object,
  showValidationResults: PropTypes.bool.isRequired,
  setShowValidationResults: PropTypes.func.isRequired,
  handleLongSeqs: PropTypes.string.isRequired,
  setHandleLongSeqs: PropTypes.func.isRequired,
  similarityData: PropTypes.object,
  methods: PropTypes.object,
};
