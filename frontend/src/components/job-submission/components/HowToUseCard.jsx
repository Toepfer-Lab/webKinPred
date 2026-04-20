// src/components/HowToUseCard.js

import React, { useState } from 'react';
import { Card, Row, Col, Alert, Button, Table } from 'react-bootstrap';
import { BoxArrowInDown, Bullseye, CloudUpload, Cpu, Github, ChevronDown, ChevronUp, Speedometer2 } from 'react-bootstrap-icons';
import '../../../styles/components/HowToUseCard.css';


const BENCHMARK_DATA = [
  { method: 'DLKcat',    uncached: '32 s',          cached: 'N/A'         },
  { method: 'CatPred',   uncached: '14 min 0 s',    cached: '23 s'        },
  { method: 'EITLEM',    uncached: '18 min 13 s',   cached: '4 min 11 s'  },
  { method: 'TurNup',    uncached: '19 min 36 s',   cached: '3 min 48 s'  },
  { method: 'CataPro',   uncached: '25 min 9 s',    cached: '41 s'        },
  { method: 'UniKP',     uncached: '33 min 46 s',   cached: '4 min 7 s'   },
  { method: 'KinForm-L', uncached: '54 min 38 s',   cached: '37 s'        },
  { method: 'KinForm-H', uncached: '56 min 10 s',   cached: '36 s'        },
];

export default function HowToUseCard({ methods = {} }) {
  const [showBenchmark, setShowBenchmark] = useState(false);
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
        <h4 className="text-center mb-3 mt-2">Input Data Format</h4>
        <p className="fmtt-subtitle">Required CSV columns for each format</p>
        <div className="fmttable">
          {/* Column headers */}
          <div className="fmtt-row fmtt-head">
            <div className="fmtt-label-cell" />
            <div className="fmtt-cell fmtt-col-head">
              <span className="fmtt-format-name">Single-Substrate</span>
              <span className="fmtt-models">DLKcat · EITLEM · UniKP · KinForm-H · KinForm-L · CataPro · CatPred (K<sub>M</sub>)</span>
            </div>
            <div className="fmtt-cell fmtt-col-head">
              <span className="fmtt-format-name">Multi-Substrate</span>
              <span className="fmtt-models">CatPred (k<sub>cat</sub> only)</span>
            </div>
            <div className="fmtt-cell fmtt-col-head">
              <span className="fmtt-format-name">Full Reaction</span>
              <span className="fmtt-models">TurNup</span>
            </div>
          </div>

          {/* Protein Sequence */}
          <div className="fmtt-row">
            <div className="fmtt-label-cell">
              <code className="fmtt-label">Protein Sequence</code>
            </div>
            <div className="fmtt-cell fmtt-present">Full amino-acid sequence</div>
            <div className="fmtt-cell fmtt-present">Full amino-acid sequence</div>
            <div className="fmtt-cell fmtt-present">Full amino-acid sequence</div>
          </div>

          {/* Substrate (singular) */}
          <div className="fmtt-row">
            <div className="fmtt-label-cell">
              <code className="fmtt-label">Substrate</code>
            </div>
            <div className="fmtt-cell fmtt-present"><code>SMILES</code> or <code>InChI</code> — one per row</div>
            <div className="fmtt-cell fmtt-present">Co-substrates joined with <code>.</code> <span className="fmtt-example">e.g. CC(=O)O.O</span></div>
            <div className="fmtt-cell fmtt-absent"><span className="fmtt-not-required">not applicable</span></div>
          </div>

          {/* Substrates (plural) */}
          <div className="fmtt-row">
            <div className="fmtt-label-cell">
              <code className="fmtt-label">Substrates</code>
            </div>
            <div className="fmtt-cell fmtt-absent"><span className="fmtt-not-required">not applicable</span></div>
            <div className="fmtt-cell fmtt-absent"><span className="fmtt-not-required">not applicable</span></div>
            <div className="fmtt-cell fmtt-present">Semicolon-separated <code>SMILES</code> or <code>InChI</code></div>
          </div>

          {/* Products */}
          <div className="fmtt-row">
            <div className="fmtt-label-cell">
              <code className="fmtt-label">Products</code>
            </div>
            <div className="fmtt-cell fmtt-absent"><span className="fmtt-not-required">not applicable</span></div>
            <div className="fmtt-cell fmtt-absent"><span className="fmtt-not-required">not applicable</span></div>
            <div className="fmtt-cell fmtt-present">Semicolon-separated <code>SMILES</code> or <code>InChI</code></div>
          </div>
        </div>

        {/* ── Timing Benchmark ── */}
        <div className="benchmark-section mt-3">
          <button
            className="benchmark-toggle"
            onClick={() => setShowBenchmark(!showBenchmark)}
            aria-expanded={showBenchmark}
          >
            <span className="benchmark-toggle-left">
              <Speedometer2 size={13} className="benchmark-toggle-icon" />
              Runtime Benchmark
            </span>
            {showBenchmark ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>

          {showBenchmark && (
            <div className="benchmark-content">
              <p className="benchmark-intro">
                Protein language model (PLM) embeddings are cached on the server. A sequence computed once is reused for all future jobs.
                Therefore, <strong>compute time scales with the number of unique protein sequences</strong>, not the number of rows.
              </p>
              <p className="benchmark-conditions">
                Benchmark conditions: <span className="benchmark-cond-value">1,000 reactions</span> ·{' '}
                <span className="benchmark-cond-value">100 unique proteins</span> ·{' '}
                <span className="benchmark-cond-value">avg. length 400 aa</span>
              </p>
              <Table size="sm" className="benchmark-table mb-0">
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>PLM embeddings not cached</th>
                    <th>PLM embeddings cached</th>
                  </tr>
                </thead>
                <tbody>
                  {BENCHMARK_DATA.map(({ method, uncached, cached }) => (
                    <tr key={method}>
                      <td className="benchmark-method">{method}</td>
                      <td>{uncached ?? '—'}</td>
                      <td className={cached === 'N/A' ? 'benchmark-na' : cached ? '' : 'benchmark-empty'}>{cached ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>
          )}
        </div>

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

          <Button
            href="/templates/full_reaction_template.csv"
            download
            className="btn btn-custom-subtle"
          >
            <BoxArrowInDown className="me-2" />
            Full-Reaction Template
          </Button>
        </div>
      </Card.Body>
    </Card>
  );
}
