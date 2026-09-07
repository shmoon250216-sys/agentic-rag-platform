const appShell = document.querySelector("#appShell");
const sidebar = document.querySelector("#sidebar");
const inspector = document.querySelector("#inspector");
const inspectorTitle = document.querySelector("#inspectorTitle");
const drawerBackdrop = document.querySelector("#drawerBackdrop");
const chatLog = document.querySelector("#chatLog");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const sendBtn = document.querySelector("#sendBtn");
const composerStatus = document.querySelector("#composerStatus");
const routeBadge = document.querySelector("#routeBadge");
const sessionText = document.querySelector("#sessionText");
const routeItems = Array.from(document.querySelectorAll(".route-item"));
const navItems = Array.from(document.querySelectorAll(".nav-item"));
const panelViews = Array.from(document.querySelectorAll(".panel-view"));

const tokenInput = document.querySelector("#tokenInput");
const userInput = document.querySelector("#userInput");
const userSummary = document.querySelector("#userSummary");
const newSessionBtn = document.querySelector("#newSessionBtn");
const refreshSessionsBtn = document.querySelector("#refreshSessionsBtn");
const sessionList = document.querySelector("#sessionList");

const docTitleInput = document.querySelector("#docTitleInput");
const docContentInput = document.querySelector("#docContentInput");
const addDocBtn = document.querySelector("#addDocBtn");
const refreshDocumentsBtn = document.querySelector("#refreshDocumentsBtn");
const docStatus = document.querySelector("#docStatus");
const docList = document.querySelector("#docList");
const docNavCount = document.querySelector("#docNavCount");
const ingestModeButtons = Array.from(document.querySelectorAll("[data-ingest-mode]"));
const ingestViews = Array.from(document.querySelectorAll("[data-ingest-view]"));
const docFileInput = document.querySelector("#docFileInput");
const uploadZone = document.querySelector("#uploadZone");
const selectedFileName = document.querySelector("#selectedFileName");
const selectedFileMeta = document.querySelector("#selectedFileMeta");
const fileTitleInput = document.querySelector("#fileTitleInput");
const uploadDocBtn = document.querySelector("#uploadDocBtn");

const memoryTypeInput = document.querySelector("#memoryTypeInput");
const memoryContentInput = document.querySelector("#memoryContentInput");
const addMemoryBtn = document.querySelector("#addMemoryBtn");
const refreshMemoriesBtn = document.querySelector("#refreshMemoriesBtn");
const memoryStatus = document.querySelector("#memoryStatus");
const memoryList = document.querySelector("#memoryList");
const memoryNavCount = document.querySelector("#memoryNavCount");

const healthDot = document.querySelector("#healthDot");
const systemStatus = document.querySelector("#systemStatus");
const systemInfoList = document.querySelector("#systemInfoList");
const qualitySummary = document.querySelector("#qualitySummary");
const qualityMetrics = document.querySelector("#qualityMetrics");
const cacheStats = document.querySelector("#cacheStats");
const toolList = document.querySelector("#toolList");
const refreshSystemBtn = document.querySelector("#refreshSystemBtn");
const clearCacheBtn = document.querySelector("#clearCacheBtn");

let sessionId = null;
let activeRequestController = null;
let selectedDocumentFile = null;

function authHeaders() {
  const token = tokenInput.value.trim() || "dev-token";
  return { Authorization: `Bearer ${token}` };
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function createEmptyState(text) {
  return createElement("div", "empty-state", text);
}

function clearWelcome() {
  chatLog.querySelector("[data-welcome]")?.remove();
}

function appendMessage(role, text, extra = {}) {
  clearWelcome();
  const article = createElement("article", `message ${role}`);
  const avatar = createElement("div", "avatar", role === "user" ? "你" : "AI");
  const bubble = createElement("div", "bubble");
  const content = createElement("div", "markdown-body");
  renderMarkdown(content, text, { autoFormat: role === "assistant" });
  bubble.appendChild(content);
  appendResponseDetails(bubble, extra);
  article.append(avatar, bubble);
  chatLog.appendChild(article);
  scrollChatToBottom();
}

function appendStreamingAssistant() {
  clearWelcome();
  const article = createElement("article", "message assistant");
  const avatar = createElement("div", "avatar", "AI");
  const bubble = createElement("div", "bubble");
  const content = createElement("div", "markdown-body");
  renderMarkdown(content, "", { autoFormat: true });
  bubble.appendChild(content);
  article.append(avatar, bubble);
  chatLog.appendChild(article);
  scrollChatToBottom();
  return { bubble, content, rawText: "" };
}

function appendResponseDetails(bubble, extra = {}) {
  if (!extra.route && !extra.sources?.length && !extra.toolResult) return;

  const details = createElement("details", "response-details");
  const summary = createElement("summary", "", "查看执行详情");
  details.appendChild(summary);

  if (extra.route) {
    const meta = createElement("div", "meta");
    meta.appendChild(createPill(`分支：${routeLabel(extra.route.route)}`));
    meta.appendChild(createPill(`置信度：${Math.round(extra.route.confidence * 100)}%`));
    if (extra.route.reason) meta.appendChild(createPill(extra.route.reason));
    details.appendChild(meta);
  }

  if (extra.sources?.length) {
    const sources = createElement("div", "sources");
    extra.sources.forEach((source) => {
      const item = createElement("div", "source");
      item.append(
        createElement("strong", "", `${source.title} · 相关度 ${Math.round(source.score * 100)}%`),
        createElement("span", "", source.snippet),
      );
      sources.appendChild(item);
    });
    details.appendChild(sources);
  }

  if (extra.toolResult) {
    const sources = createElement("div", "sources");
    const item = createElement("div", "source");
    item.append(
      createElement("strong", "", "工具执行结果"),
      createElement("span", "", JSON.stringify(extra.toolResult, null, 2)),
    );
    sources.appendChild(item);
    details.appendChild(sources);
  }

  bubble.appendChild(details);
}

function createPill(text) {
  return createElement("span", "pill", text);
}

function renderMarkdown(container, markdown, options = {}) {
  const fragment = document.createDocumentFragment();
  const normalizedMarkdown = options.autoFormat
    ? normalizeAssistantText(markdown)
    : String(markdown || "");
  const lines = normalizedMarkdown.replace(/\r\n/g, "\n").split("\n");
  let index = 0;

  while (index < lines.length) {
    const trimmed = lines[index].trim();
    if (!trimmed) {
      index += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      const pre = document.createElement("pre");
      const code = createElement("code", "", codeLines.join("\n"));
      pre.appendChild(code);
      fragment.appendChild(pre);
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      const level = Math.min(headingMatch[1].length + 2, 5);
      const heading = document.createElement(`h${level}`);
      appendInlineText(heading, headingMatch[2]);
      fragment.appendChild(heading);
      index += 1;
      continue;
    }

    if (/^>\s?/.test(trimmed)) {
      const quote = document.createElement("blockquote");
      const quoteLines = [];
      while (index < lines.length && /^>\s?/.test(lines[index].trim())) {
        quoteLines.push(lines[index].trim().replace(/^>\s?/, ""));
        index += 1;
      }
      appendInlineText(quote, quoteLines.join("\n"));
      fragment.appendChild(quote);
      continue;
    }

    const unorderedMatch = trimmed.match(/^[-*+]\s+(.+)$/);
    const orderedMatch = trimmed.match(/^(?:\d+|[一二三四五六七八九十]+)[.)、．]\s*(.+)$/);
    if (unorderedMatch || orderedMatch) {
      const list = document.createElement(unorderedMatch ? "ul" : "ol");
      const listPattern = unorderedMatch
        ? /^[-*+]\s+(.+)$/
        : /^(?:\d+|[一二三四五六七八九十]+)[.)、．]\s*(.+)$/;
      while (index < lines.length) {
        const itemMatch = lines[index].trim().match(listPattern);
        if (!itemMatch) break;
        const item = document.createElement("li");
        appendInlineText(item, itemMatch[1]);
        list.appendChild(item);
        index += 1;
      }
      fragment.appendChild(list);
      continue;
    }

    const paragraphLines = [];
    while (index < lines.length && !isMarkdownBoundary(lines[index])) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    const paragraph = document.createElement("p");
    appendInlineText(paragraph, paragraphLines.join("\n"));
    fragment.appendChild(paragraph);
  }

  if (!fragment.childNodes.length) {
    const paragraph = createElement("p", "is-empty", " ");
    fragment.appendChild(paragraph);
  }
  container.replaceChildren(fragment);
}

function normalizeAssistantText(markdown) {
  const text = String(markdown || "").trim();
  if (!text || text.includes("```")) return text;
  if (text.includes("\n") && hasStructuredMarkdown(text)) return text;

  const withChineseBreaks = text
    .replace(/([。！？；;.!?])\s*((?:\d{1,2}|[一二三四五六七八九十]+)[、.．)]\s*)/g, "$1\n$2")
    .replace(
      /([。！？；;.!?])\s*(首先|其次|然后|接着|另外|同时|最后|总结|结论|因此|例如|注意)[:：，,]/g,
      "$1\n$2：",
    );

  return withChineseBreaks.includes("\n")
    ? withChineseBreaks
    : splitLongChineseParagraph(withChineseBreaks);
}

function hasStructuredMarkdown(text) {
  return (
    /\n\s*\n/.test(text) ||
    /^#{1,3}\s+/m.test(text) ||
    /^[-*+]\s+/m.test(text) ||
    /^(?:\d+|[一二三四五六七八九十]+)[.)、．]\s*/m.test(text)
  );
}

function splitLongChineseParagraph(text) {
  if (text.length <= 120) return text;
  const sentences = text.match(/[^。！？；.!?;]+[。！？；.!?;]?/g) || [text];
  const paragraphs = [];
  let current = "";
  sentences.forEach((sentence) => {
    if (current && `${current}${sentence}`.length > 95) {
      paragraphs.push(current);
      current = sentence;
    } else {
      current += sentence;
    }
  });
  if (current) paragraphs.push(current);
  return paragraphs.join("\n\n");
}

function isMarkdownBoundary(line) {
  const trimmed = line.trim();
  return (
    !trimmed ||
    trimmed.startsWith("```") ||
    /^(#{1,3})\s+/.test(trimmed) ||
    /^>\s?/.test(trimmed) ||
    /^[-*+]\s+/.test(trimmed) ||
    /^(?:\d+|[一二三四五六七八九十]+)[.)、．]\s*/.test(trimmed)
  );
}

function appendInlineText(parent, text) {
  const value = String(text || "");
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let cursor = 0;
  value.replace(pattern, (match, _token, offset) => {
    if (offset > cursor) parent.appendChild(document.createTextNode(value.slice(cursor, offset)));
    if (match.startsWith("`")) {
      parent.appendChild(createElement("code", "", match.slice(1, -1)));
    } else {
      parent.appendChild(createElement("strong", "", match.slice(2, -2)));
    }
    cursor = offset + match.length;
    return match;
  });
  if (cursor < value.length) parent.appendChild(document.createTextNode(value.slice(cursor)));
}

function routeLabel(route) {
  return {
    rag: "知识检索",
    tool: "工具调用",
    plan: "任务规划",
    chat: "普通对话",
    fallback: "安全兜底",
  }[route] || "等待提问";
}

function memoryTypeLabel(type) {
  return {
    profile: "用户画像",
    preference: "用户偏好",
    goal: "长期目标",
    project_fact: "项目事实",
    decision: "关键决策",
  }[type] || "长期记忆";
}

function setRoute(route) {
  routeBadge.textContent = routeLabel(route);
  routeItems.forEach((item) => item.classList.toggle("active", item.dataset.route === route));
}

function setBusy(isBusy) {
  sendBtn.classList.toggle("is-busy", isBusy);
  sendBtn.textContent = isBusy ? "■" : "↑";
  sendBtn.title = isBusy ? "停止生成" : "发送消息";
  sendBtn.setAttribute("aria-label", sendBtn.title);
  composerStatus.textContent = isBusy ? "Agent 正在处理请求" : "回答由模型生成，请核对重要信息。";
}

function scrollChatToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function resetChatLog() {
  chatLog.replaceChildren();
}

async function sendMessage(message) {
  if (activeRequestController) {
    activeRequestController.abort();
    return;
  }

  const userId = userInput.value.trim() || "local-user";
  const controller = new AbortController();
  activeRequestController = controller;
  appendMessage("user", message);
  const stream = appendStreamingAssistant();
  setBusy(true);

  try {
    const response = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ message, session_id: sessionId, user_id: userId }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await readError(response, "请求失败"));

    const donePayload = await consumeChatStream(response, stream);
    sessionId = donePayload.session_id;
    sessionText.textContent = `会话 ${sessionId.slice(0, 8)}`;
    renderMarkdown(stream.content, stream.rawText, { autoFormat: true });
    appendResponseDetails(stream.bubble, {
      route: donePayload.route,
      sources: donePayload.sources,
      toolResult: donePayload.tool_result,
    });
    await Promise.all([refreshSessions(), refreshMemories()]);
  } catch (error) {
    const messageText = error.name === "AbortError" ? "已停止生成。" : `请求失败：${error.message}`;
    renderMarkdown(stream.content, messageText, { autoFormat: true });
  } finally {
    if (activeRequestController === controller) activeRequestController = null;
    setBusy(false);
    messageInput.focus();
  }
}

async function consumeChatStream(response, stream) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let donePayload = null;

  const processParts = (parts) => {
    parts.forEach((part) => {
      const event = parseSseEvent(part);
      if (!event) return;
      if (event.event === "route") setRoute(event.data.route);
      if (event.event === "token") {
        stream.rawText += event.data.text;
        renderMarkdown(stream.content, stream.rawText, { autoFormat: true });
        scrollChatToBottom();
      }
      if (event.event === "done") donePayload = event.data;
    });
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(/\r?\n\r?\n/);
    buffer = parts.pop() || "";
    processParts(parts);
  }
  buffer += decoder.decode();
  if (buffer.trim()) processParts([buffer]);
  if (!donePayload) throw new Error("流式响应没有返回完成事件");
  return donePayload;
}

function parseSseEvent(rawEvent) {
  const lines = rawEvent.split(/\r?\n/);
  let eventName = "message";
  const dataLines = [];
  lines.forEach((line) => {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  });
  if (!dataLines.length) return null;
  return { event: eventName, data: JSON.parse(dataLines.join("\n")) };
}

async function readError(response, fallback) {
  try {
    const payload = await response.json();
    return payload.error?.message || payload.detail || fallback;
  } catch (_error) {
    return fallback;
  }
}

async function fetchJson(url, options = {}, fallback = "请求失败") {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await readError(response, fallback));
  return response.json();
}

async function refreshSessions() {
  const userId = userInput.value.trim() || "local-user";
  try {
    const payload = await fetchJson(
      `/api/v1/sessions?user_id=${encodeURIComponent(userId)}`,
      { headers: authHeaders() },
      "历史会话加载失败",
    );
    sessionList.replaceChildren();
    if (!payload.sessions.length) {
      sessionList.appendChild(createEmptyState("还没有历史对话"));
      return;
    }
    payload.sessions.forEach((session) => {
      const item = createElement("button", "session-card");
      item.type = "button";
      item.classList.toggle("active", session.session_id === sessionId);
      if (session.session_id === sessionId) sessionText.textContent = session.title;
      item.append(
        createElement("strong", "", session.title || `会话 ${session.session_id.slice(0, 8)}`),
        createElement(
          "small",
          "",
          `${session.message_count} 条消息 · ${new Date(session.updated_at).toLocaleString()}`,
        ),
      );
      item.title = session.title;
      item.addEventListener("click", () => loadSession(session.session_id, session.title));
      sessionList.appendChild(item);
    });
  } catch (error) {
    sessionList.replaceChildren(createEmptyState(`加载失败：${error.message}`));
  }
}

async function loadSession(selectedSessionId, selectedTitle) {
  try {
    const payload = await fetchJson(
      `/api/v1/sessions/${encodeURIComponent(selectedSessionId)}/messages`,
      { headers: authHeaders() },
      "消息加载失败",
    );
    sessionId = selectedSessionId;
    sessionText.textContent = selectedTitle || `会话 ${sessionId.slice(0, 8)}`;
    setRoute(null);
    resetChatLog();
    payload.messages.forEach((message) => {
      appendMessage(message.role === "assistant" ? "assistant" : "user", message.content);
    });
    await refreshSessions();
    closeDrawers();
  } catch (error) {
    appendMessage("assistant", `消息加载失败：${error.message}`);
  }
}

async function refreshDocuments() {
  try {
    const payload = await fetchJson(
      "/api/v1/documents",
      { headers: authHeaders() },
      "知识库加载失败",
    );
    docStatus.textContent = `${payload.documents.length} 篇文档 · ${payload.total_chunks} 个片段`;
    docNavCount.textContent = String(payload.documents.length);
    docList.replaceChildren();
    if (!payload.documents.length) {
      docList.appendChild(createEmptyState("知识库中还没有文档"));
      return;
    }
    payload.documents.forEach((documentItem) => {
      const card = createElement("article", "resource-card");
      const head = createElement("div", "resource-head");
      const deleteBtn = createElement("button", "delete-btn", "×");
      deleteBtn.type = "button";
      deleteBtn.title = "删除文档";
      deleteBtn.setAttribute("aria-label", `删除文档 ${documentItem.title}`);
      deleteBtn.addEventListener("click", () => deleteDocument(documentItem));
      head.append(createElement("strong", "", documentItem.title), deleteBtn);
      card.append(
        head,
        createElement(
          "small",
          "",
          `${documentItem.chunk_count} 个片段 · ${documentItem.content_length} 字符`,
        ),
      );
      docList.appendChild(card);
    });
  } catch (error) {
    docStatus.textContent = `加载失败：${error.message}`;
    docList.replaceChildren(createEmptyState("暂时无法读取知识库"));
  }
}

async function addDocument() {
  const title = docTitleInput.value.trim();
  const content = docContentInput.value.trim();
  if (!title || content.length < 20) {
    docStatus.textContent = "请填写标题，并输入至少 20 个字的内容。";
    return;
  }
  addDocBtn.disabled = true;
  addDocBtn.textContent = "正在处理";
  try {
    const payload = await fetchJson(
      "/api/v1/documents",
      {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ title, content }),
      },
      "文档添加失败",
    );
    docContentInput.value = "";
    docStatus.textContent = `已收录：${payload.document.title}`;
    await refreshDocuments();
  } catch (error) {
    docStatus.textContent = `添加失败：${error.message}`;
  } finally {
    addDocBtn.disabled = false;
    addDocBtn.textContent = "加入知识库";
  }
}

function setIngestMode(mode) {
  ingestModeButtons.forEach((button) => {
    const active = button.dataset.ingestMode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  ingestViews.forEach((view) => {
    view.hidden = view.dataset.ingestView !== mode;
  });
}

function selectDocumentFile(file) {
  if (!file) {
    selectedDocumentFile = null;
    selectedFileName.textContent = "选择或拖入 PDF、DOCX";
    selectedFileMeta.textContent = "单个文件最大 10MB";
    uploadZone.classList.remove("has-file");
    uploadDocBtn.disabled = true;
    return;
  }

  const extension = file.name.toLowerCase().split(".").pop();
  if (!["pdf", "docx"].includes(extension)) {
    docStatus.textContent = "文件格式不支持，请选择 PDF 或 DOCX。";
    docFileInput.value = "";
    selectDocumentFile(null);
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    docStatus.textContent = "文件超过 10MB，请拆分后再上传。";
    docFileInput.value = "";
    selectDocumentFile(null);
    return;
  }

  selectedDocumentFile = file;
  selectedFileName.textContent = file.name;
  selectedFileMeta.textContent = `${formatFileSize(file.size)} · ${extension.toUpperCase()}`;
  uploadZone.classList.add("has-file");
  uploadDocBtn.disabled = false;
  docStatus.textContent = "文件已选择，等待解析。";
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function uploadDocument() {
  if (!selectedDocumentFile) {
    docStatus.textContent = "请先选择 PDF 或 DOCX 文件。";
    return;
  }

  uploadDocBtn.disabled = true;
  uploadDocBtn.textContent = "正在解析";
  const formData = new FormData();
  formData.append("file", selectedDocumentFile);
  const customTitle = fileTitleInput.value.trim();
  if (customTitle) formData.append("title", customTitle);

  try {
    const payload = await fetchJson(
      "/api/v1/documents/upload",
      {
        method: "POST",
        headers: authHeaders(),
        body: formData,
      },
      "文件上传失败",
    );
    docStatus.textContent = `已解析并收录：${payload.document.title}`;
    docFileInput.value = "";
    fileTitleInput.value = "";
    selectDocumentFile(null);
    await refreshDocuments();
  } catch (error) {
    docStatus.textContent = `上传失败：${error.message}`;
  } finally {
    uploadDocBtn.textContent = "解析并加入知识库";
    uploadDocBtn.disabled = !selectedDocumentFile;
  }
}

async function deleteDocument(documentItem) {
  if (!window.confirm(`确定删除“${documentItem.title}”吗？`)) return;
  try {
    const payload = await fetchJson(
      `/api/v1/documents/${encodeURIComponent(documentItem.doc_id)}`,
      { method: "DELETE", headers: authHeaders() },
      "文档删除失败",
    );
    if (!payload.deleted) throw new Error("文档不存在或已经删除");
    docStatus.textContent = `已删除：${documentItem.title}`;
    await refreshDocuments();
  } catch (error) {
    docStatus.textContent = `删除失败：${error.message}`;
  }
}

async function refreshMemories() {
  const userId = userInput.value.trim() || "local-user";
  try {
    const payload = await fetchJson(
      `/api/v1/users/${encodeURIComponent(userId)}/memories`,
      { headers: authHeaders() },
      "长期记忆加载失败",
    );
    memoryStatus.textContent = `${payload.memories.length} 条长期记忆`;
    memoryNavCount.textContent = String(payload.memories.length);
    memoryList.replaceChildren();
    if (!payload.memories.length) {
      memoryList.appendChild(createEmptyState("当前用户还没有长期记忆"));
      return;
    }
    payload.memories.forEach((memory) => {
      const card = createElement("article", "resource-card");
      const foot = createElement("div", "resource-foot");
      const deleteBtn = createElement("button", "delete-btn", "×");
      deleteBtn.type = "button";
      deleteBtn.title = "删除记忆";
      deleteBtn.setAttribute("aria-label", "删除这条长期记忆");
      deleteBtn.addEventListener("click", () => deleteMemory(memory.memory_id));
      foot.append(
        createElement(
          "span",
          "memory-tag",
          `${memoryTypeLabel(memory.memory_type)} · 重要度 ${Math.round(memory.importance * 100)}%`,
        ),
        deleteBtn,
      );
      card.append(createElement("p", "", memory.content), foot);
      memoryList.appendChild(card);
    });
  } catch (error) {
    memoryStatus.textContent = `加载失败：${error.message}`;
    memoryList.replaceChildren(createEmptyState("暂时无法读取长期记忆"));
  }
}

async function addMemory() {
  const userId = userInput.value.trim() || "local-user";
  const content = memoryContentInput.value.trim();
  if (content.length < 4) {
    memoryStatus.textContent = "请输入至少 4 个字的记忆内容。";
    return;
  }
  addMemoryBtn.disabled = true;
  addMemoryBtn.textContent = "正在保存";
  try {
    const payload = await fetchJson(
      `/api/v1/users/${encodeURIComponent(userId)}/memories`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ memory_type: memoryTypeInput.value, content, importance: 0.7 }),
      },
      "长期记忆添加失败",
    );
    memoryContentInput.value = "";
    memoryStatus.textContent = `已保存：${memoryTypeLabel(payload.memory.memory_type)}`;
    await refreshMemories();
  } catch (error) {
    memoryStatus.textContent = `保存失败：${error.message}`;
  } finally {
    addMemoryBtn.disabled = false;
    addMemoryBtn.textContent = "加入长期记忆";
  }
}

async function deleteMemory(memoryId) {
  if (!window.confirm("确定删除这条长期记忆吗？")) return;
  const userId = userInput.value.trim() || "local-user";
  try {
    const payload = await fetchJson(
      `/api/v1/users/${encodeURIComponent(userId)}/memories/${memoryId}`,
      { method: "DELETE", headers: authHeaders() },
      "长期记忆删除失败",
    );
    if (!payload.deleted) throw new Error("记忆不存在或已经删除");
    memoryStatus.textContent = "长期记忆已删除";
    await refreshMemories();
  } catch (error) {
    memoryStatus.textContent = `删除失败：${error.message}`;
  }
}

function appendInfoItem(container, label, value, className = "info-item") {
  const item = createElement("div", className);
  item.append(createElement("span", "", label), createElement("strong", "", String(value)));
  container.appendChild(item);
}

async function checkHealth() {
  try {
    await fetchJson("/health", {}, "服务不可用");
    healthDot.className = "health-dot online";
    return true;
  } catch (_error) {
    healthDot.className = "health-dot offline";
    return false;
  }
}

async function refreshSystemPanel() {
  systemStatus.textContent = "正在读取运行状态";
  const headers = authHeaders();
  const [health, info, quality, cache, tools] = await Promise.allSettled([
    fetchJson("/health", {}, "服务不可用"),
    fetchJson("/api/v1/system/info", { headers }, "配置读取失败"),
    fetchJson("/api/v1/quality/gate", { headers }, "质量检测失败"),
    fetchJson("/api/v1/cache/stats", { headers }, "缓存统计失败"),
    fetchJson("/api/v1/tools", { headers }, "工具清单读取失败"),
  ]);

  const healthy = health.status === "fulfilled";
  healthDot.className = `health-dot ${healthy ? "online" : "offline"}`;
  systemStatus.textContent = healthy ? "服务运行正常" : "服务连接异常";
  renderSystemInfo(info);
  renderQualityGate(quality);
  renderCacheStats(cache);
  renderTools(tools);
}

function renderSystemInfo(result) {
  systemInfoList.replaceChildren();
  if (result.status !== "fulfilled") {
    systemInfoList.appendChild(createEmptyState(result.reason.message));
    return;
  }
  const info = result.value;
  appendInfoItem(systemInfoList, "LLM", `${info.llm.provider} / ${info.llm.model}`);
  appendInfoItem(systemInfoList, "RAG", info.rag.backend);
  appendInfoItem(systemInfoList, "Embedding", info.embedding.provider);
  appendInfoItem(systemInfoList, "会话存储", info.memory.session_store);
}

function renderQualityGate(result) {
  qualityMetrics.replaceChildren();
  if (result.status !== "fulfilled") {
    qualitySummary.className = "quality-badge failed";
    qualitySummary.textContent = "检测失败";
    qualityMetrics.appendChild(createEmptyState(result.reason.message));
    return;
  }
  const report = result.value;
  qualitySummary.className = `quality-badge ${report.passed ? "passed" : "failed"}`;
  qualitySummary.textContent = report.passed ? "通过" : "未通过";
  const labels = {
    total_cases: "评测样本",
    route_accuracy: "路由准确率",
    rag_hit_rate: "RAG 命中率",
    tool_accuracy: "工具准确率",
    average_latency_ms: "平均延迟",
  };
  report.checks.forEach((check) => {
    const item = createElement("div", `metric-item ${check.passed ? "passed" : "failed"}`);
    let actual = check.actual ?? "--";
    if (check.metric.includes("accuracy") || check.metric.includes("rate")) {
      actual = `${Math.round(Number(actual) * 100)}%`;
    } else if (check.metric.includes("latency")) {
      actual = `${actual} ms`;
    }
    item.append(
      createElement("span", "", labels[check.metric] || check.metric),
      createElement("strong", "", actual),
    );
    qualityMetrics.appendChild(item);
  });
}

function renderCacheStats(result) {
  cacheStats.replaceChildren();
  if (result.status !== "fulfilled") {
    cacheStats.appendChild(createEmptyState(result.reason.message));
    return;
  }
  appendInfoItem(cacheStats, "检索缓存", `${result.value.retrieval.size} 条`);
  appendInfoItem(cacheStats, "模型缓存", `${result.value.llm_response.size} 条`);
  appendInfoItem(cacheStats, "检索命中", result.value.retrieval.hits);
  appendInfoItem(cacheStats, "模型命中", result.value.llm_response.hits);
}

function renderTools(result) {
  toolList.replaceChildren();
  if (result.status !== "fulfilled") {
    toolList.appendChild(createEmptyState(result.reason.message));
    return;
  }
  result.value.tools.forEach((tool) => {
    const card = createElement("article", "resource-card");
    card.append(createElement("strong", "", tool.name), createElement("p", "", tool.description));
    toolList.appendChild(card);
  });
}

async function clearCache() {
  clearCacheBtn.disabled = true;
  try {
    await fetchJson(
      "/api/v1/cache",
      { method: "DELETE", headers: authHeaders() },
      "缓存清理失败",
    );
    const cache = await Promise.allSettled([
      fetchJson("/api/v1/cache/stats", { headers: authHeaders() }, "缓存统计失败"),
    ]);
    renderCacheStats(cache[0]);
  } catch (error) {
    cacheStats.replaceChildren(createEmptyState(error.message));
  } finally {
    clearCacheBtn.disabled = false;
  }
}

function openPanel(name) {
  navItems.forEach((item) => item.classList.toggle("active", item.dataset.panel === name));
  appShell.classList.remove("sidebar-open");
  if (name === "chat") {
    closeInspector();
    return;
  }
  const titles = { knowledge: "知识库", memory: "长期记忆", system: "系统状态" };
  inspectorTitle.textContent = titles[name] || "工作区";
  panelViews.forEach((view) => {
    view.hidden = view.dataset.view !== name;
  });
  appShell.classList.add("inspector-open");
  inspector.setAttribute("aria-hidden", "false");
  if (name === "knowledge") refreshDocuments();
  if (name === "memory") refreshMemories();
  if (name === "system") refreshSystemPanel();
}

function closeInspector() {
  appShell.classList.remove("inspector-open");
  inspector.setAttribute("aria-hidden", "true");
  navItems.forEach((item) => item.classList.toggle("active", item.dataset.panel === "chat"));
}

function closeDrawers() {
  appShell.classList.remove("sidebar-open");
  if (window.innerWidth <= 1120) closeInspector();
}

function autoResizeComposer() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 160)}px`;
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (activeRequestController) {
    activeRequestController.abort();
    return;
  }
  const message = messageInput.value.trim();
  if (!message) return;
  messageInput.value = "";
  autoResizeComposer();
  sendMessage(message);
});

messageInput.addEventListener("input", autoResizeComposer);
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

navItems.forEach((item) => item.addEventListener("click", () => openPanel(item.dataset.panel)));
newSessionBtn.addEventListener("click", () => {
  if (activeRequestController) activeRequestController.abort();
  sessionId = null;
  sessionText.textContent = "新会话";
  setRoute(null);
  resetChatLog();
  appendMessage("assistant", "新的会话已经准备好。你可以直接开始提问。 ");
  refreshSessions();
  closeDrawers();
});

document.querySelector("#openSidebarBtn").addEventListener("click", () => {
  appShell.classList.add("sidebar-open");
});
document.querySelector("#closeSidebarBtn").addEventListener("click", closeDrawers);
document.querySelector("#closeInspectorBtn").addEventListener("click", closeInspector);
document.querySelector("#openSettingsBtn").addEventListener("click", () => openPanel("system"));
drawerBackdrop.addEventListener("click", closeDrawers);

refreshSessionsBtn.addEventListener("click", refreshSessions);
refreshDocumentsBtn.addEventListener("click", refreshDocuments);
addDocBtn.addEventListener("click", addDocument);
uploadDocBtn.addEventListener("click", uploadDocument);
ingestModeButtons.forEach((button) => {
  button.addEventListener("click", () => setIngestMode(button.dataset.ingestMode));
});
docFileInput.addEventListener("change", () => selectDocumentFile(docFileInput.files[0]));
uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadZone.classList.add("dragging");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragging"));
uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadZone.classList.remove("dragging");
  selectDocumentFile(event.dataTransfer.files[0]);
});
refreshMemoriesBtn.addEventListener("click", refreshMemories);
addMemoryBtn.addEventListener("click", addMemory);
refreshSystemBtn.addEventListener("click", refreshSystemPanel);
clearCacheBtn.addEventListener("click", clearCache);
userInput.addEventListener("change", () => {
  const userId = userInput.value.trim() || "local-user";
  userSummary.textContent = userId;
  refreshSessions();
  refreshMemories();
});
tokenInput.addEventListener("change", () => {
  refreshSessions();
  refreshDocuments();
  refreshMemories();
});

checkHealth();
refreshDocuments();
refreshSessions();
refreshMemories();
autoResizeComposer();
