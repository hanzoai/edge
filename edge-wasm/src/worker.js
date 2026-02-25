// Hanzo Edge WASM -- Web Worker for off-main-thread inference.
//
// Usage from the main thread:
//
//   const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
//
//   worker.postMessage({ type: 'init', data: { modelBytes, tokenizerBytes } });
//   worker.onmessage = (e) => { ... };
//
// Messages IN  (main -> worker):
//   { type: 'init',     data: { modelBytes: Uint8Array, tokenizerBytes: Uint8Array } }
//   { type: 'generate', data: { prompt: string, maxTokens: number, temperature: number } }
//   { type: 'stream',   data: { prompt: string, maxTokens: number, temperature: number } }
//   { type: 'reset' }
//
// Messages OUT (worker -> main):
//   { type: 'ready',    version: string, device: string }
//   { type: 'token',    text: string }          // streaming only
//   { type: 'result',   text: string }
//   { type: 'error',    message: string }
//   { type: 'reset_ok' }

import init, { EdgeModel, get_version, get_device_info } from '../pkg/edge_wasm.js';

let model = null;

self.onmessage = async (e) => {
  const { type, data } = e.data;

  try {
    switch (type) {
      case 'init': {
        await init();
        const { modelBytes, tokenizerBytes } = data;
        model = new EdgeModel(
          new Uint8Array(modelBytes),
          new Uint8Array(tokenizerBytes),
        );
        self.postMessage({
          type: 'ready',
          version: get_version(),
          device: get_device_info(),
        });
        break;
      }

      case 'generate': {
        if (!model) throw new Error('model not loaded');
        const { prompt, maxTokens, temperature } = data;
        const text = model.generate(prompt, maxTokens, temperature);
        self.postMessage({ type: 'result', text });
        break;
      }

      case 'stream': {
        if (!model) throw new Error('model not loaded');
        const { prompt, maxTokens, temperature } = data;
        const text = model.generate_stream(
          prompt,
          maxTokens,
          temperature,
          (token) => {
            self.postMessage({ type: 'token', text: token });
          },
        );
        self.postMessage({ type: 'result', text });
        break;
      }

      case 'reset': {
        if (model) model.reset();
        self.postMessage({ type: 'reset_ok' });
        break;
      }

      default:
        throw new Error(`unknown message type: ${type}`);
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message || String(err) });
  }
};
