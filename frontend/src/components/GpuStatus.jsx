import React, { useEffect, useState } from 'react';
import apiClient from './appClient';


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

  const schedule = 'Available daily 5 PM – 8 AM GMT · 24h on weekends';

  if (!status.configured || !status.online) {
    return (
      <div className="gpu-status-bar gpu-status-bar--cpu" data-tooltip={schedule}>
        <span className="gpu-status-bar__dot" />
        <span className="gpu-status-bar__label">CPU Mode</span>
        <span className="gpu-status-bar__sep">·</span>
        <span className="gpu-status-bar__detail">GPU acceleration available</span>
      </div>
    );
  }

  const now = new Date();
  const cutoff = new Date(Date.UTC(
    now.getUTCFullYear(),
    now.getUTCMonth(),
    now.getUTCDate() + (now.getUTCHours() >= 8 ? 1 : 0),
    8, 0, 0, 0
  ));
  const msLeft = cutoff - now;
  const hLeft = Math.floor(msLeft / 3600000);
  const mLeft = Math.floor((msLeft % 3600000) / 60000);
  const remaining = hLeft > 0 ? `${hLeft}h ${mLeft}m` : `${mLeft}m`;

  return (
    <div className="gpu-status-bar gpu-status-bar--online" data-tooltip={schedule}>
      <span className="gpu-status-bar__dot" />
      <span className="gpu-status-bar__label">GPU Available</span>
      <span className="gpu-status-bar__sep">·</span>
      <span className="gpu-status-bar__detail">{remaining} remaining</span>
    </div>
  );
}
