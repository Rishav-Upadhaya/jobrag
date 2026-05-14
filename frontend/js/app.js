const chatContainer = document.getElementById("chat-container");
const historyList = document.getElementById("history-list");
const input = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const welcomeScreen = document.getElementById("welcome-screen");

let chatHistory = [];
let isStreaming = false;

const STAGE_CLASS = {
  classifying: "stage-classifying",
  retrieving: "stage-retrieving",
  synthesizing: "stage-synthesizing",
  judging: "stage-judging",
};

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  autoResizeInput();
  input.focus();
});

function bindEvents() {
  sendBtn.addEventListener("click", () => sendQuery(input.value));

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendQuery(input.value);
    }
  });

  input.addEventListener("input", autoResizeInput);

  document.querySelectorAll(".suggestion-chip").forEach((chip) => {
    chip.addEventListener("click", () => sendQuery(chip.textContent || ""));
  });
}

function autoResizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function setStreamingState(active) {
  isStreaming = active;
  sendBtn.disabled = active;
}

function hideWelcome() {
  if (welcomeScreen) {
    welcomeScreen.style.display = "none";
  }
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function appendUserMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message user";

  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;

  wrapper.appendChild(content);
  chatContainer.appendChild(wrapper);
}

function createAssistantMessage() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";

  const shell = document.createElement("div");
  shell.className = "assistant-shell";

  const status = document.createElement("div");
  status.className = `status-indicator ${STAGE_CLASS.classifying}`;
  status.innerHTML = `
    <span class="dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>
    <span class="status-text">Analyzing your query...</span>
  `;

  shell.appendChild(status);
  wrapper.appendChild(shell);
  chatContainer.appendChild(wrapper);

  return { wrapper, shell, status };
}

function updateStatus(statusEl, stage, message) {
  statusEl.classList.remove(...Object.values(STAGE_CLASS));
  const stageClass = STAGE_CLASS[stage] || STAGE_CLASS.classifying;
  statusEl.classList.add(stageClass);

  const statusText = statusEl.querySelector(".status-text");
  if (statusText) {
    statusText.textContent = message || "Working...";
  }
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInlineMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>");
}

function renderMarkdown(raw) {
  const safe = escapeHtml(raw || "");
  const lines = safe.split(/\r?\n/);
  const chunks = [];
  let listBuffer = [];
  let paraBuffer = [];

  const flushList = () => {
    if (listBuffer.length > 0) {
      const items = listBuffer.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("");
      chunks.push(`<ul>${items}</ul>`);
      listBuffer = [];
    }
  };

  const flushParagraph = () => {
    if (paraBuffer.length > 0) {
      chunks.push(`<p>${renderInlineMarkdown(paraBuffer.join(" "))}</p>`);
      paraBuffer = [];
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed) {
      flushList();
      flushParagraph();
      continue;
    }

    if (trimmed.startsWith("## ")) {
      flushList();
      flushParagraph();
      chunks.push(`<h3>${renderInlineMarkdown(trimmed.slice(3))}</h3>`);
      continue;
    }

    if (trimmed.startsWith("# ")) {
      flushList();
      flushParagraph();
      chunks.push(`<h2>${renderInlineMarkdown(trimmed.slice(2))}</h2>`);
      continue;
    }

    if (trimmed.startsWith("- ")) {
      flushParagraph();
      listBuffer.push(trimmed.slice(2));
      continue;
    }

    flushList();
    paraBuffer.push(trimmed);
  }

  flushList();
  flushParagraph();

  return chunks.join("");
}

function renderJobCard(source) {
  const title = escapeHtml(source.job_title || "Untitled role");
  const company = escapeHtml(source.company_name || "Unknown company");
  const level = escapeHtml(source.job_level || "N/A");
  const location = escapeHtml(source.job_location || "N/A");
  const excerpt = escapeHtml(source.matched_chunk || "No excerpt available");
  const score = Number(source.relevance_score || 0);
  const scorePct = Math.max(0, Math.min(100, Math.round(score * 100)));

  return `
    <article class="job-card">
      <div class="job-header">
        <div class="job-title">${title}</div>
        <div class="company-name">${company}</div>
      </div>
      <div class="job-meta">
        <span class="meta-pill">${level}</span>
        <span class="meta-pill">${location}</span>
      </div>
      <div class="job-excerpt">${excerpt}</div>
      <div class="job-score">
        <div class="job-score-label">Relevance ${scorePct}%</div>
        <div class="job-score-track">
          <div class="job-score-bar" style="width:${scorePct}%"></div>
        </div>
      </div>
    </article>
  `;
}

function renderClarificationCard(question) {
  const safeQuestion = renderMarkdown(question || "What role, skill, level, or location should I search for?");

  return `
    <article class="clarification-card">
      <div class="clarification-label">Clarification needed</div>
      <div class="clarification-question">${safeQuestion}</div>
      <div class="clarification-hint">Reply with the missing detail and I’ll search again.</div>
    </article>
  `;
}

function toggleSources(button) {
  const panelId = button.getAttribute("aria-controls");
  if (!panelId) return;

  const panel = document.getElementById(panelId);
  if (!panel) return;

  const isOpen = panel.classList.toggle("open");
  const count = Number(button.dataset.count || "0");

  button.textContent = isOpen
    ? `📋 Hide sources ▲`
    : `📋 View ${count} sources ▼`;
}

function addHistoryItem(queryText) {
  const item = document.createElement("button");
  item.type = "button";
  item.className = "history-item";
  item.textContent = queryText;
  item.addEventListener("click", () => sendQuery(queryText));

  historyList.prepend(item);
}

function parseSseEvent(block) {
  console.debug("[parseSseEvent] Received block (first 200 chars):", block.substring(0, 200));
  const lines = block.split("\n");
  let eventName = "message";
  let data = "{}";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
      continue;
    }

    if (line.startsWith("data:")) {
      data = line.slice(5).trim();
    }
  }

  let parsedData = {};
  try {
    parsedData = JSON.parse(data);
    console.debug("[parseSseEvent] Parsed successfully. Event:", eventName, "Data keys:", Object.keys(parsedData));
  } catch (error) {
    console.error("[parseSseEvent] JSON.parse failed for data:", data.substring(0, 100), "Error:", error.message);
    parsedData = {};
  }

  return { eventName, data: parsedData };
}

async function streamQuery(query, assistantRef) {
  console.log("[streamQuery] Starting stream for query:", query);
  setStreamingState(true);

  let finalized = false;
  let fullAnswer = "";
  let responseState = {
    sources: [],
    judge: null,
    intent: "valid",
    clarification_question: "",
    latency_ms: 0,
  };

  try {
    console.log("[streamQuery] Building conversation history from chat");
    // Build conversation history from chatHistory (excluding current message which was already added)
    const conversationHistory = chatHistory.slice(0, -1).map(msg => ({
      role: msg.role,
      content: msg.content
    }));

    console.log("[streamQuery] Sending with", conversationHistory.length, "prior messages");
    console.log("[streamQuery] Fetching /api/query/stream");
    const response = await fetch("/api/query/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({ 
        query, 
        conversation_history: conversationHistory 
      }),
    });

    console.log("[streamQuery] Fetch response status:", response.status, "ok:", response.ok);

    if (!response.ok || !response.body) {
      throw new Error(`Streaming request failed (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let chunkCount = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log("[streamQuery] Reader finished. Total chunks read:", chunkCount);
        break;
      }

      chunkCount++;
      const decodedChunk = decoder.decode(value, { stream: true });
      console.debug("[streamQuery] Chunk", chunkCount, '- received', decodedChunk.length, 'bytes');
      buffer += decodedChunk;

      const splits = buffer.split("\n\n");
      console.debug("[streamQuery] After split, got", splits.length, "parts. Last part (incomplete):", splits[splits.length-1].substring(0, 50));
      
      buffer = splits.pop() || "";

      for (const chunk of splits) {
        if (!chunk.trim()) {
          console.debug("[streamQuery] Skipping empty chunk");
          continue;
        }

        console.debug("[streamQuery] Processing SSE block");
        const { eventName, data } = parseSseEvent(chunk);
        console.log("[streamQuery] Event handler check - eventName:", eventName);

        if (eventName === "status") {
          console.log("[streamQuery] Handling status event:", data.stage);
          updateStatus(assistantRef.status, data.stage, data.message);
          scrollToBottom();
          continue;
        }

        if (eventName === "answer") {
          console.log("[streamQuery] Handling answer event, text length:", data.text?.length || 0);
          fullAnswer = data.text || "";
          assistantRef.status.remove();
          const answerEl = document.createElement("div");
          answerEl.className = "message-content";
          answerEl.innerHTML = renderMarkdown(data.text || "");
          assistantRef.answerEl = answerEl;
          assistantRef.shell.appendChild(answerEl);
          scrollToBottom();
          continue;
        }

        if (eventName === "sources") {
          console.log("[streamQuery] Handling sources event, count:", Array.isArray(data.sources) ? data.sources.length : 0);
          const sources = Array.isArray(data.sources) ? data.sources : [];
          responseState = {
            sources,
            judge: data.judge || null,
            intent: data.intent || "valid",
            clarification_question: data.clarification_question || "",
            latency_ms: Number(data.latency_ms || 0),
          };

          if (responseState.intent === "vague") {
            const clarificationQuestion =
              data.clarification_question ||
              fullAnswer ||
              "What role, skill, level, or location should I search for?";

            fullAnswer = clarificationQuestion;
            responseState.sources = [];
            responseState.judge = null;
            responseState.clarification_question = clarificationQuestion;

            if (assistantRef.answerEl) {
              assistantRef.answerEl.classList.add("clarification-message");
              assistantRef.answerEl.innerHTML = renderClarificationCard(clarificationQuestion);
            }

            scrollToBottom();
            continue;
          }

          const verdict = String(data.judge?.verdict || "fail").toLowerCase();
          const score = Number(data.judge?.score || 0);
          const badgeClass = verdict === "pass" ? "pass" : score >= 0.5 ? "warn" : "fail";

          const judgeBadge = document.createElement("div");
          judgeBadge.className = `judge-badge ${badgeClass}`;
          judgeBadge.textContent = `${verdict} · ${(score * 100).toFixed(0)}%`;
          assistantRef.judgeBadge = judgeBadge;
          assistantRef.shell.appendChild(judgeBadge);

          const panelId = `sources-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
          const toggle = document.createElement("button");
          toggle.type = "button";
          toggle.className = "sources-toggle";
          toggle.dataset.count = String(sources.length);
          toggle.setAttribute("aria-controls", panelId);
          toggle.textContent = `📋 View ${sources.length} sources ▼`;
          toggle.addEventListener("click", () => toggleSources(toggle));
          assistantRef.sourcesToggle = toggle;
          assistantRef.shell.appendChild(toggle);

          const panel = document.createElement("div");
          panel.className = "sources-panel";
          panel.id = panelId;
          panel.innerHTML = sources.map((source) => renderJobCard(source)).join("");
          assistantRef.sourcesPanel = panel;
          assistantRef.shell.appendChild(panel);

          scrollToBottom();
          continue;
        }

        if (eventName === "done") {
          console.log("[streamQuery] Handling done event");
          finalized = true;
          setStreamingState(false);
          scrollToBottom();
        }
      }
    }
  } catch (error) {
    console.error("[streamQuery] Caught error:", error.message, error.stack);
    if (assistantRef.status && assistantRef.status.isConnected) {
      assistantRef.status.remove();
    }

    const errEl = document.createElement("div");
    errEl.className = "message-content error";
    errEl.textContent = "Error: Could not stream response. Please try again.";
    assistantRef.shell.appendChild(errEl);
  } finally {
    if (!finalized) {
      setStreamingState(false);
    }

    // Add assistant response to conversation history
    chatHistory.push({
      role: "assistant",
      content: fullAnswer,
      sources: responseState.sources,
      judge: responseState.judge,
      intent: responseState.intent,
      clarification_question: responseState.clarification_question,
      latency_ms: responseState.latency_ms,
    });

    addHistoryItem(query);
    scrollToBottom();
  }
}

function sendQuery(queryText) {
  if (isStreaming || !queryText || !queryText.trim()) {
    return;
  }

  const query = queryText.trim();
  hideWelcome();
  appendUserMessage(query);

  chatHistory.push({ role: "user", content: query });

  const assistantRef = createAssistantMessage();

  input.value = "";
  autoResizeInput();
  scrollToBottom();
  streamQuery(query, assistantRef);
}
