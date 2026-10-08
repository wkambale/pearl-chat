// Pearl-Chat interactive client logic

document.addEventListener('DOMContentLoaded', () => {
  // DOM element references
  const chatMessages = document.getElementById('chat-messages');
  const chatForm = document.getElementById('chat-form');
  const promptInput = document.getElementById('prompt-input');
  const sendButton = document.getElementById('send-button');
  const stopButton = document.getElementById('stop-button');
  const clearChatButton = document.getElementById('clear-chat-button');

  const tempSlider = document.getElementById('temp-slider');
  const tempValue = document.getElementById('temp-value');
  const topPSlider = document.getElementById('top-p-slider');
  const topPValue = document.getElementById('top-p-value');
  const topKSlider = document.getElementById('top-k-slider');
  const topKValue = document.getElementById('top-k-value');
  const maxTokensSlider = document.getElementById('max-tokens-slider');
  const maxTokensValue = document.getElementById('max-tokens-value');

  const modelSourceInput = document.getElementById('model-source-input');
  const loadModelButton = document.getElementById('load-model-button');
  const modelStatusText = document.getElementById('model-status-text');
  const paramCountText = document.getElementById('param-count-text');
  const vocabSizeText = document.getElementById('vocab-size-text');
  const contextLenText = document.getElementById('context-len-text');
  const backendText = document.getElementById('backend-text');

  const metricTokens = document.getElementById('metric-tokens');
  const metricSpeed = document.getElementById('metric-speed');

  const themeToggleButton = document.getElementById('theme-toggle-button');
  const themeToggleLabel = document.getElementById('theme-toggle-label');
  const themeIconSun = document.getElementById('theme-icon-sun');
  const themeIconMoon = document.getElementById('theme-icon-moon');

  const promptChips = document.querySelectorAll('.prompt-chip');

  let activeAbortController = null;
  let isGenerating = false;

  // Theme management
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('pearlchat-theme', theme);
    if (theme === 'light') {
      if (themeToggleLabel) themeToggleLabel.textContent = 'Dark mode';
      if (themeIconSun) themeIconSun.classList.add('hidden');
      if (themeIconMoon) themeIconMoon.classList.remove('hidden');
    } else {
      if (themeToggleLabel) themeToggleLabel.textContent = 'Light mode';
      if (themeIconSun) themeIconSun.classList.remove('hidden');
      if (themeIconMoon) themeIconMoon.classList.add('hidden');
    }
  }

  const savedTheme = localStorage.getItem('pearlchat-theme') || 'dark';
  applyTheme(savedTheme);

  if (themeToggleButton) {
    themeToggleButton.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      applyTheme(current === 'dark' ? 'light' : 'dark');
    });
  }

  // Sync sliders
  tempSlider.addEventListener('input', () => {
    tempValue.textContent = parseFloat(tempSlider.value).toFixed(2);
  });
  topPSlider.addEventListener('input', () => {
    topPValue.textContent = parseFloat(topPSlider.value).toFixed(2);
  });
  topKSlider.addEventListener('input', () => {
    topKValue.textContent = topKSlider.value;
  });
  maxTokensSlider.addEventListener('input', () => {
    maxTokensValue.textContent = maxTokensSlider.value;
  });

  // Prompt suggestions
  promptChips.forEach(chip => {
    chip.addEventListener('click', () => {
      promptInput.value = chip.dataset.prompt;
      promptInput.focus();
      autoResizeTextarea();
    });
  });

  // Auto-resize textarea
  function autoResizeTextarea() {
    promptInput.style.height = 'auto';
    promptInput.style.height = Math.min(promptInput.scrollHeight, 140) + 'px';
  }
  promptInput.addEventListener('input', autoResizeTextarea);

  promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && promptInput.value.trim()) {
        chatForm.dispatchEvent(new Event('submit'));
      }
    }
  });

  // Fetch model info on startup
  async function fetchModelInfo() {
    try {
      const response = await fetch('/api/info');
      if (!response.ok) return;
      const data = await response.json();
      paramCountText.textContent = data.parameters ? Number(data.parameters).toLocaleString() : 'N/A';
      vocabSizeText.textContent = data.vocab_size ? Number(data.vocab_size).toLocaleString() : '8,192';
      contextLenText.textContent = data.context_length ? `${data.context_length} tokens` : '128 tokens';
      backendText.textContent = data.backend || 'JAX CPU';
      modelStatusText.textContent = data.status || 'Ready';
      if (data.source) {
        modelSourceInput.value = data.source;
      }
    } catch (err) {
      console.warn('Failed to fetch model info:', err);
    }
  }

  fetchModelInfo();

  // Load model button
  loadModelButton.addEventListener('click', async () => {
    const source = modelSourceInput.value.trim();
    if (!source) return;

    loadModelButton.disabled = true;
    modelStatusText.textContent = 'Loading...';
    try {
      const response = await fetch('/api/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source }),
      });
      const data = await response.json();
      if (response.ok) {
        modelStatusText.textContent = 'Ready';
        fetchModelInfo();
      } else {
        modelStatusText.textContent = 'Failed';
        alert('Error loading model: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      modelStatusText.textContent = 'Error';
      alert('Network error connecting to inference server');
    } finally {
      loadModelButton.disabled = false;
    }
  });

  // Clear conversation
  clearChatButton.addEventListener('click', () => {
    chatMessages.innerHTML = '';
    metricTokens.textContent = '0';
    metricSpeed.textContent = '0.0 tok/s';
  });

  // Append user or assistant message
  function appendMessage(sender, text, isUser = false) {
    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ${isUser ? 'user-wrapper' : 'assistant-wrapper'}`;

    const senderDiv = document.createElement('div');
    senderDiv.className = 'message-sender';
    senderDiv.textContent = sender;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;

    wrapper.appendChild(senderDiv);
    wrapper.appendChild(contentDiv);
    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return contentDiv;
  }

  // Handle form submission and streaming
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = promptInput.value.trim();
    if (!prompt || isGenerating) return;

    appendMessage('User', prompt, true);
    promptInput.value = '';
    autoResizeTextarea();

    const assistantContentEl = appendMessage('Assistant', '');
    const cursor = document.createElement('span');
    cursor.className = 'streaming-cursor';
    assistantContentEl.appendChild(cursor);

    isGenerating = true;
    sendButton.classList.add('hidden');
    stopButton.classList.remove('hidden');

    activeAbortController = new AbortController();

    const startTime = performance.now();
    let tokenCount = 0;

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          temperature: parseFloat(tempSlider.value),
          top_p: parseFloat(topPSlider.value),
          top_k: parseInt(topKSlider.value, 10),
          max_tokens: parseInt(maxTokensSlider.value, 10),
        }),
        signal: activeAbortController.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Inference error (${response.status}): ${errorText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      let streamFinished = false;
      while (!streamFinished) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6);
            if (dataStr === '[DONE]') {
              streamFinished = true;
              break;
            }
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.token) {
                cursor.remove();
                assistantContentEl.textContent += parsed.token;
                assistantContentEl.appendChild(cursor);
                tokenCount++;
                const elapsedSec = (performance.now() - startTime) / 1000;
                metricTokens.textContent = tokenCount.toString();
                metricSpeed.textContent = `${(tokenCount / Math.max(elapsedSec, 0.01)).toFixed(1)} tok/s`;
                chatMessages.scrollTop = chatMessages.scrollHeight;
              }
            } catch (jsonErr) {
              // Non-JSON line or chunk
            }
          }
        }

        if (streamFinished) {
          try {
            await reader.cancel();
          } catch (cancelErr) {
            // Reader cancel handled
          }
          break;
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        cursor.remove();
        assistantContentEl.textContent += ' (Stopped by user)';
      } else {
        cursor.remove();
        assistantContentEl.textContent += ` [Error: ${err.message}]`;
      }
    } finally {
      cursor.remove();
      isGenerating = false;
      activeAbortController = null;
      sendButton.classList.remove('hidden');
      stopButton.classList.add('hidden');
      promptInput.disabled = false;
      promptInput.focus();
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  });

  stopButton.addEventListener('click', () => {
    if (activeAbortController) {
      activeAbortController.abort();
    }
  });
});
