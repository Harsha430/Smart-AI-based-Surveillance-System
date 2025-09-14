import { useEffect, useRef, useState } from 'react';

export function useAlerts(url = '/ws/alerts') {
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const host = window.location.host;
      const full = url.startsWith('ws') ? url : `${protocol}://${host}${url}`;
      const ws = new WebSocket(full);
      wsRef.current = ws;
      ws.onopen = () => !cancelled && setConnected(true);
      ws.onclose = () => {
        if (!cancelled) {
          setConnected(false);
          setTimeout(connect, 2000);
        }
      };
      ws.onerror = () => ws.close();
      ws.onmessage = ev => {
        try {
          const data = JSON.parse(ev.data);
          data._receivedAt = Date.now();
          setAlerts(prev => {
            const next = [...prev, data];
            return next.slice(-300); // cap
          });
        } catch (e) {
          // ignore
        }
      };
    };
    connect();
    return () => {
      cancelled = true;
      wsRef.current && wsRef.current.close();
    };
  }, [url]);

  return { alerts, connected };
}

