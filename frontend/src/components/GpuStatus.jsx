import React, { useEffect, useState } from 'react';
import { Table, Container, Row, Col, Alert } from 'react-bootstrap';
import { ChevronDown, ChevronUp, Speedometer2 } from 'react-bootstrap-icons';
import apiClient from './appClient';
import '../styles/components/GpuStatus.css';

const BENCHMARK_DATA = [
  { method: 'DLKcat',    uncachedCpu: '32 s',         uncachedGpu: '', cached: 'N/A'        },
  { method: 'CatPred',   uncachedCpu: '14 min 0 s',   uncachedGpu: '', cached: '23 s'       },
  { method: 'EITLEM',    uncachedCpu: '18 min 13 s',  uncachedGpu: '', cached: '4 min 11 s' },
  { method: 'TurNup',    uncachedCpu: '19 min 36 s',  uncachedGpu: '', cached: '3 min 48 s' },
  { method: 'CataPro',   uncachedCpu: '25 min 9 s',   uncachedGpu: '', cached: '41 s'       },
  { method: 'UniKP',     uncachedCpu: '33 min 46 s',  uncachedGpu: '', cached: '4 min 7 s'  },
  { method: 'KinForm-L', uncachedCpu: '54 min 38 s',  uncachedGpu: '', cached: '37 s'       },
  { method: 'KinForm-H', uncachedCpu: '56 min 10 s',  uncachedGpu: '', cached: '36 s'       },
];

export default function GpuStatus({ layout = 'home' }) {
  const [status, setStatus] = useState(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const fetchStatus = async () => {
      try {
        const { data } = await apiClient.get('/v1/gpu/status/');
        if (!cancelled) setStatus(data || { configured: false, online: false, mode: 'cpu' });
      } catch (_) {
        if (!cancelled) setStatus({ configured: false, online: false, mode: 'cpu' });
      }
    };
    fetchStatus();
    const timer = setInterval(fetchStatus, 15000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  if (!status) return null;

  const isOnline = status.configured && status.online;
  const schedule = 'Daily 5 PM – 8 AM GMT · 24h on weekends';

  let remaining = null;
  if (isOnline) {
    const now = new Date();
    const cutoff = new Date(Date.UTC(
      now.getUTCFullYear(), now.getUTCMonth(),
      now.getUTCDate() + (now.getUTCHours() >= 8 ? 1 : 0),
      8, 0, 0, 0
    ));
    const msLeft = cutoff - now;
    const hLeft = Math.floor(msLeft / 3600000);
    const mLeft = Math.floor((msLeft % 3600000) / 60000);
    remaining = hLeft > 0 ? `${hLeft}h ${mLeft}m` : `${mLeft}m`;
  }

  return (
    <section className="gpu-status-shell">
      <Container className="gpu-status-container">
        <Row className="justify-content-center">
          <Col md={10} lg={layout === 'track' ? 9 : undefined}>
            <Alert variant="info" className="d-flex align-items-start gpu-status-alert mb-0">
              <Speedometer2 size={24} className="me-3 mt-1 gpu-status-icon" />
              <div className="gpu-status-content">
                <div className="gpu-status-line">
                  <strong className="gpu-status-title">{isOnline ? 'GPU Available' : 'CPU Mode'}</strong>
                  <span className="gpu-status-value">
                    {isOnline ? `${remaining} remaining` : 'GPU currently unavailable'}
                  </span>
                </div>

                <p className="gpu-status-meta mb-2">
                  <strong>GPU Window</strong>
                  <span>Daily 5 PM - 8 AM GMT · 24h on weekends</span>
                </p>

                <button
                  type="button"
                  className="gpu-status-benchmark-toggle mb-1"
                  onClick={() => setOpen((v) => !v)}
                  aria-expanded={open}
                >
                  <strong>Runtime Benchmark</strong>
                  {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </button>

                {open && (
                  <>
                    <p className="gpu-status-intro">
                      PLM embeddings are cached server-side. Once a sequence is computed, it is reused across future jobs.
                      Compute time scales mainly with unique proteins, not total rows.
                    </p>
                    <p className="gpu-status-conditions">
                      <strong>Conditions:</strong>
                      <span className="gpu-status-cond-value">1,000 reactions</span>
                      <span className="gpu-status-cond-sep">·</span>
                      <span className="gpu-status-cond-value">100 unique proteins</span>
                      <span className="gpu-status-cond-sep">·</span>
                      <span className="gpu-status-cond-value">avg. 400 aa</span>
                    </p>

                    <Table responsive size="sm" className="benchmark-table mb-0">
                      <colgroup>
                        <col className="benchmark-col-method" />
                        <col className="benchmark-col-gpu" />
                        <col className="benchmark-col-cpu" />
                        <col className="benchmark-col-cached" />
                      </colgroup>
                      <thead>
                        <tr>
                          <th rowSpan={2} className="benchmark-th-method">Method</th>
                          <th colSpan={2} className="benchmark-th-group">PLM embeddings not cached</th>
                          <th className="benchmark-th-group benchmark-th-cached">PLM embeddings cached</th>
                        </tr>
                        <tr>
                          <th className="benchmark-th-sub">GPU</th>
                          <th className="benchmark-th-sub">CPU</th>
                          <th className="benchmark-th-sub benchmark-th-sub-placeholder" aria-hidden="true">&nbsp;</th>
                        </tr>
                      </thead>
                      <tbody>
                        {BENCHMARK_DATA.map(({ method, uncachedGpu, uncachedCpu, cached }) => (
                          <tr key={method}>
                            <td className="benchmark-method">{method}</td>
                            <td className={`benchmark-time ${uncachedGpu ? '' : 'benchmark-empty'}`}>{uncachedGpu || '—'}</td>
                            <td className="benchmark-time">{uncachedCpu ?? '—'}</td>
                            <td className={`benchmark-time ${cached === 'N/A' ? 'benchmark-na' : cached ? '' : 'benchmark-empty'}`}>
                              {cached ?? '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </>
                )}
              </div>
            </Alert>
          </Col>
        </Row>
      </Container>
    </section>
  );
}
