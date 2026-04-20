import React, { useEffect, useState } from 'react';
import { OverlayTrigger, Tooltip } from 'react-bootstrap';
import apiClient from './appClient';

function formatGb(value) {
  if (value === null || value === undefined || value === '') return null;
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return `${n.toFixed(1)} GB`;
}

function StatusBadge({ className, tooltipText, children }) {
  return (
    <OverlayTrigger
      placement="bottom"
      overlay={<Tooltip className="gpu-status-tooltip">{tooltipText}</Tooltip>}
    >
      <span className={`gpu-status-badge ${className}`}>
        {children}
      </span>
    </OverlayTrigger>
  );
}

export default function GpuStatus() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const fetchStatus = async () => {
      try {
        const { data } = await apiClient.get('/v1/gpu/status/');
        if (!cancelled) setStatus(data || { configured: false, online: false, mode: 'cpu' });
      } catch (_) {
        if (!cancelled) {
          setStatus({ configured: false, online: false, mode: 'cpu' });
        }
      }
    };

    fetchStatus();
    const timer = setInterval(fetchStatus, 15000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  if (!status) return null;

  if (!status.configured || !status.online) {
    const tooltip = status.configured
      ? 'GPU service is offline. GPU is expected to be available between 5 PM – 8 AM GMT.'
      : 'Running in CPU mode. GPU is expected to be available between 5 PM – 8 AM GMT.';
    return (
      <StatusBadge className="gpu-status-cpu" tooltipText={tooltip}>
        <span className="gpu-status-dot" />
        CPU mode
      </StatusBadge>
    );
  }

  const gpuName = status.gpu_name || 'GPU';
  const free = formatGb(status.free_vram_gb);
  const freeLabel = free ? ` · ${free} free` : '';
  return (
    <StatusBadge className="gpu-status-online" tooltipText={`GPU online: ${gpuName}${freeLabel}`}>
      <span className="gpu-status-dot" />
      {gpuName}{freeLabel}
    </StatusBadge>
  );
}
