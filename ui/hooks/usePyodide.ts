import { useState, useEffect } from "react";

declare global {
  interface Window {
    loadPyodide?: () => Promise<any>;
  }
}

const PYODIDE_VERSION = "0.25.0";

export default function usePyodide() {
  const [pyodide, setPyodide] = useState<any>(null);

  useEffect(() => {
    const loadPyodide = async () => {
      if (typeof window !== 'undefined' && !window.loadPyodide) {
        const script = document.createElement('script');
        script.src = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.js`;
        script.async = true;
        script.onload = async () => {
          const loadedPyodide = await (window as any).loadPyodide({
            indexURL: `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`,
          });
          setPyodide(loadedPyodide);
        };
        document.body.appendChild(script);
      } else if (typeof window !== 'undefined' && window.loadPyodide) {
        const loadedPyodide = await (window as any).loadPyodide({
          indexURL: `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`,
        });
        setPyodide(loadedPyodide);
      }
    };

    loadPyodide();
  }, []);

  return { pyodide };
}