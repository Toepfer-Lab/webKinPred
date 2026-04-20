import React, { useEffect, useState } from 'react';
import apiClient from './appClient';

function formatGb(value) {
  if (value === null || value === undefined || value === '') return null;
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return `${n.toFixed(1)} GB`;
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
    return (
      <div className="gpu-status-bar gpu-status-bar--cpu">
        <span className="gpu-status-bar__dot" />
        <span className="gpu-status-bar__label">CPU Mode</span>
        <span className="gpu-status-bar__sep">·</span>
        <span className="gpu-status-bar__detail">
          GPU acceleration available 5 PM – 8 AM GMT
        </span>
      </div>
    );
  }

  const gpuName = status.gpu_name || 'GPU';
  const free = formatGb(status.free_vram_gb);
  return (
    <div className="gpu-status-bar gpu-status-bar--online">
      <span className="gpu-status-bar__dot" />
      <span className="gpu-status-bar__label">GPU Accelerated</span>
      <span className="gpu-status-bar__sep">·</span>
      <span className="gpu-status-bar__detail">{gpuName}{free ? ` · ${free} free` : ''}</span>
    </div>
  );
}
