// src/components/HowToUseCard.js

import React from 'react';
import { Card, Row, Col, Alert, ListGroup, Button } from 'react-bootstrap';
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

        <h4 className="text-center mb-4">Input Data Format</h4>
        <Row>
          <Col md={6} className="mb-3">
            <Card className="h-100 format-card">
              <Card.Body>
                <Card.Title>Single-Substrate Models</Card.Title>
                <Card.Subtitle className="mb-2 text-white-70">
                  DLKcat, EITLEM, UniKP, KinForm-H, KinForm-L, CataPro
                </Card.Subtitle>
                <ListGroup variant="flush">
                  <ListGroup.Item>
                    <span className="csv-col">Protein Sequence</span> — Full amino-acid sequence.
                  </ListGroup.Item>
                  <ListGroup.Item>
                    <span className="csv-col">Substrate</span> — One <code>SMILES</code> or <code>InChI</code> string.
                  </ListGroup.Item>
                </ListGroup>
              </Card.Body>
            </Card>
          </Col>
          <Col md={6} className="mb-3">
            <Card className="h-100 format-card">
              <Card.Body>
                <Card.Title>Multi-Substrate Model</Card.Title>
                <Card.Subtitle className="mb-2 text-white-70">TurNup</Card.Subtitle>
                <ListGroup variant="flush">
                  <ListGroup.Item>
                    <span className="csv-col">Protein Sequence</span> — Full amino-acid sequence.
                  </ListGroup.Item>
                  <ListGroup.Item>
                    <span className="csv-col">Substrates</span> — Semicolon-separated list of <code>SMILES</code> or <code>InChI</code>.
                  </ListGroup.Item>
                  <ListGroup.Item>
                    <span className="csv-col">Products</span> — Semicolon-separated list of <code>SMILES</code> or <code>InChI</code>.
                  </ListGroup.Item>
                </ListGroup>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        <p className="text-center text-white-70">
            Multi-substrate CSVs can also be used for K<sub>M</sub> predictions. Each entry in the 'Substrates' column will receive its own K<sub>M</sub> value (separated by semicolons).
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

        <hr className="my-4" />
        <h4 className="text-center mb-4">Available Prediction Methods</h4>
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
      </Card.Body>
    </Card>
  );
}
