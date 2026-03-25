// src/components/HowToUseCard.js

import React from 'react';
import { Card, Row, Col, Alert, Button } from 'react-bootstrap';
import { BoxArrowInDown, Bullseye, CloudUpload, Cpu, Github } from 'react-bootstrap-icons';
import '../../../styles/components/HowToUseCard.css';


export default function HowToUseCard({ methods = {} }) {
  const methodEntries = Object.entries(methods).sort(([, a], [, b]) =>
    (a.displayName || '').localeCompare(b.displayName || '')
  );
  const targetLabel = {
    kcat: 'kcat',
    Km: 'Km',
    'kcat/Km': 'kcat/Km',
  };

  return (
    <Card className="section-container how-to-use-card mb-4">
      <Card.Header as="h3" className="text-center">
        How to Use This Tool
      </Card.Header>
      <Card.Body>
        <p className="lead text-center mb-4">
          Predict kinetic parameters (k<sub>cat</sub>, K<sub>M</sub>, and k<sub>cat</sub>/K<sub>M</sub>) for enzyme-catalysed reactions using various machine learning models.
        </p>
        <Alert variant="info" className="d-flex align-items-center">
          <Bullseye size={24} className="me-3" />
          <div>
            Ticking <strong>“Prefer experimental data”</strong> will first search BRENDA, SABIO-RK, and UniProt for known values. If found, these are used instead of model predictions.
          </div>
        </Alert>

        <Row className="text-center">
          <Col md={4} className="step-col">
            <div className="step-icon"><Bullseye size={30} /></div>
            <h5>Step 1: Select Prediction</h5>
            <p>Choose one or more targets: k<sub>cat</sub>, K<sub>M</sub>, and/or k<sub>cat</sub>/K<sub>M</sub>.</p>
          </Col>
          <Col md={4} className="step-col">
            <div className="step-icon"><CloudUpload size={30} /></div>
            <h5>Step 2: Upload Data</h5>
            <p>Provide your reaction data by uploading a formatted CSV file.</p>
          </Col>
          <Col md={4} className="step-col">
            <div className="step-icon"><Cpu size={30} /></div>
            <h5>Step 3: Choose Method</h5>
            <p>Select your desired prediction model(s) after optional validation.</p>
          </Col>
        </Row>

        <hr className="my-4" />

        <h4 className="text-center mb-4">Available Predictors</h4>
        <Row className="g-2 method-cards-grid">
          {methodEntries.map(([key, details]) => (
            <Col key={key} xs={12} sm={6} lg={4} xl={3}>
              <Card className="method-card h-100">
                <Card.Body className="method-card-body">
                  <div className="method-head">
                    <Card.Title className="method-title mb-0">{details.displayName}</Card.Title>
                    {details.repoUrl && (
                      <a
                        href={details.repoUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="github-link"
                        title="View on GitHub"
                      >
                        <Github size={18} />
                      </a>
                    )}
                  </div>

                  <div className="method-publication">
                    {details.citationUrl ? (
                      <a
                        href={details.citationUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="publication-link"
                      >
                        <span className="publication-title">{details.publicationTitle}</span>
                      </a>
                    ) : (
                      <span className="publication-title">{details.publicationTitle}</span>
                    )}
                    {details.authors && (
                      <small className="method-authors">{details.authors}</small>
                    )}
                  </div>

                  {details.moreInfo && (
                    <div className="method-note">
                      <span className="method-note-label">Note:</span> {details.moreInfo}
                    </div>
                  )}

                  <div className="method-targets">
                    <span className="method-targets-label">Predicts</span>
                    <div className="method-target-list">
                      {(details.supports || []).map((target) => (
                        <span key={`${key}-${target}`} className="method-target-chip">
                          {targetLabel[target] || target}
                        </span>
                      ))}
                    </div>
                  </div>
                </Card.Body>
              </Card>
            </Col>
          ))}
        </Row>
        <p className="format-section-label my-2">Input Data Format</p>
        <div className="d-flex gap-3 flex-column flex-md-row">
          <div className="format-panel flex-fill">
            <div className="format-panel-title">Single-Substrate</div>
            <div className="format-panel-models">DLKcat · EITLEM · UniKP · KinForm-H · KinForm-L · CataPro</div>
            <div className="format-fields">
              <div className="format-header-row">
                <span className="format-header">Column</span>
                <span className="format-header">Expected content</span>
              </div>
              <div className="format-row">
                <span className="format-chip">Protein Sequence</span>
                <span className="format-desc">Full amino-acid sequence</span>
              </div>
              <div className="format-row">
                <span className="format-chip">Substrate</span>
                <span className="format-desc">One <code>SMILES</code> or <code>InChI</code> string</span>
              </div>
            </div>
          </div>
          <div className="format-panel flex-fill">
            <div className="format-panel-title">Multi-Substrate <span className="format-panel-model">· TurNup</span></div>
            <div className="format-panel-models">&nbsp;</div>
            <div className="format-fields">
              <div className="format-header-row">
                <span className="format-header">Column</span>
                <span className="format-header">Expected content</span>
              </div>
              <div className="format-row">
                <span className="format-chip">Protein Sequence</span>
                <span className="format-desc">Full amino-acid sequence</span>
              </div>
              <div className="format-row">
                <span className="format-chip">Substrates</span>
                <span className="format-desc">Semicolon-separated <code>SMILES</code> or <code>InChI</code></span>
              </div>
              <div className="format-row">
                <span className="format-chip">Products</span>
                <span className="format-desc">Semicolon-separated <code>SMILES</code> or <code>InChI</code></span>
              </div>
            </div>
          </div>
        </div>
        <p className="format-note mt-2">
          Multi-substrate CSVs can also be used for K<sub>M</sub> predictions. Each 'Substrates' entry receives its own K<sub>M</sub> value (semicolon-separated).
        </p>

        <hr className="my-4" />
        <h4 className="text-center mb-3">Example Templates</h4>
        <div className="d-grid gap-2 d-md-flex justify-content-md-center">
          <Button
            href="/templates/single_substrate_template.csv"
            download
            className="btn btn-custom-subtle"
          >
            <BoxArrowInDown className="me-2" />
            Single-Substrate Template
          </Button>

          <Button
            href="/templates/multi_substrate_template.csv"
            download
            className="btn btn-custom-subtle"
          >
            <BoxArrowInDown className="me-2" />
            Multi-Substrate Template
          </Button>
        </div>
      </Card.Body>
    </Card>
  );
}
