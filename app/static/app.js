const form = document.querySelector("#researchForm");
const input = document.querySelector("#queryInput");
const submit = document.querySelector("#submitButton");
const answerPanel = document.querySelector("#answerPanel");
const charCount = document.querySelector("#charCount");

input.addEventListener("input", () => { charCount.textContent = `${input.value.length} / 4000`; });
document.querySelectorAll("[data-query]").forEach(button => {
  button.addEventListener("click", () => {
    input.value = button.dataset.query;
    charCount.textContent = `${input.value.length} / 4000`;
    input.focus();
  });
});

async function checkStatus() {
  const status = document.querySelector("#systemStatus");
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Status check failed");
    const data = await response.json();
    if (data.corpus_available === false) {
      status.className = "status demo";
      status.innerHTML = "<i></i> Demo mode · case corpus not loaded";
    } else {
      status.className = "status ok";
      status.innerHTML = `<i></i> ${data.mode === "retrieval_only" ? "Local retrieval ready" : "System ready"}`;
    }
  } catch {
    status.className = "status error";
    status.innerHTML = "<i></i> System unavailable";
  }
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value == null ? "" : String(value);
  return element.innerHTML;
}

function groupByCase(results) {
  const cases = new Map();
  for (const result of results) {
    const key = result.document_id || result.file_name || result.chunk_id;
    if (!cases.has(key)) {
      cases.set(key, { ...result, passages: [], pages: new Set() });
    }
    const matchingCase = cases.get(key);
    matchingCase.passages.push(result);
    for (const page of [result.page_start, result.page_end]) {
      if (page != null) matchingCase.pages.add(page);
    }
    if (
      result.retrieval_similarity != null &&
      (matchingCase.retrieval_similarity == null ||
        result.retrieval_similarity > matchingCase.retrieval_similarity)
    ) {
      matchingCase.retrieval_similarity = result.retrieval_similarity;
    }
  }
  return [...cases.values()];
}

function normalizeCaseName(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function findCasesMentioned(question, cases) {
  const normalizedQuestion = normalizeCaseName(question);
  const namedCases = cases.filter(matchingCase => {
    const normalizedName = normalizeCaseName(matchingCase.case_name);
    return normalizedName.length >= 8 && normalizedQuestion.includes(normalizedName);
  });
  if (namedCases.length) return namedCases;

  const indexes = new Set();
  for (const match of question.matchAll(/\bcase\s+([a-e1-5])\b/gi)) {
    const reference = match[1].toLowerCase();
    indexes.add(/[1-5]/.test(reference) ? Number(reference) - 1 : reference.charCodeAt(0) - 97);
  }
  return cases.filter((_matchingCase, index) => indexes.has(index));
}

function removeCaseNamesFromQuestion(question, cases) {
  let cleaned = question
    .replace(/\bcase\s+[a-e1-5]\b/gi, " ")
    .replace(/\b(first|second|third|fourth|fifth)\s+case\b/gi, " ");
  for (const matchingCase of cases) {
    const nameTokens = normalizeCaseName(matchingCase.case_name).split(" ").filter(Boolean);
    if (nameTokens.length < 2) continue;
    const titlePattern = new RegExp(`\\b${nameTokens.join("[^a-z0-9]+")}\\b`, "gi");
    cleaned = cleaned.replace(titlePattern, " ");
  }
  return cleaned.replace(/\s+/g, " ").trim();
}

function getPassageHighlights(passages, query, limit = 3) {
  const ignoredTerms = new Set([
    "about", "after", "also", "and", "any", "are", "been", "before", "being",
    "between", "both", "but", "can", "case", "cases", "could", "did", "does",
    "court", "decide", "decided", "each", "for", "from", "have", "how", "into",
    "judgment", "judgments",
    "may", "more", "most", "not", "our", "same", "should", "some", "such",
    "say", "says", "than", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "through", "under", "was", "were", "what", "when", "where",
    "which", "while", "who", "with", "would", "your",
  ]);
  const terms = new Set(
    (query.toLowerCase().match(/[a-z0-9]{3,}/g) || [])
      .filter(term => !ignoredTerms.has(term))
  );
  const sentences = passages.flatMap((passage, passageIndex) =>
    String(passage.excerpt || "")
      .split(/(?<=[.!?])\s+(?=[A-Z0-9"'(])/)
      .map((text, sentenceIndex) => ({
        text: text.replace(/\s+/g, " ").trim(),
        passageIndex,
        sentenceIndex,
      }))
      .filter(sentence => sentence.text)
  );
  const ranked = sentences.map(sentence => {
    const sentenceTerms = new Set(sentence.text.toLowerCase().match(/[a-z0-9]{3,}/g) || []);
    return {
      ...sentence,
      matchedTerms: [...terms].filter(term => sentenceTerms.has(term)),
      score: [...terms].reduce((score, term) => score + Number(sentenceTerms.has(term)), 0),
    };
  });
  const matching = ranked.filter(sentence => sentence.score > 0);
  const bestScore = Math.max(0, ...matching.map(sentence => sentence.score));
  const usefulMatches = matching.filter(sentence =>
    sentence.score >= Math.min(2, bestScore)
  );
  const selected = (usefulMatches.length ? usefulMatches : ranked)
    .sort((left, right) =>
      right.score - left.score ||
      left.passageIndex - right.passageIndex ||
      left.sentenceIndex - right.sentenceIndex
    )
    .slice(0, limit);
  return selected.map(sentence => ({
    text: sentence.text.length > 420 ? `${sentence.text.slice(0, 417).trimEnd()}…` : sentence.text,
    passageIndex: sentence.passageIndex,
    matchedTerms: sentence.matchedTerms,
  }));
}

function getComparisonTerms(text) {
  const ignoredTerms = new Set([
    "about", "after", "also", "analyze", "analysis", "and", "any", "are",
    "assess", "assessment", "been", "before", "being", "between", "both",
    "but", "came", "can", "case", "cases", "compare", "comparison", "court",
    "could", "did", "difference", "differences", "different", "does", "each",
    "evaluate", "evaluation", "from", "have", "how", "into", "judgment",
    "judgments", "may", "more", "most", "not", "our", "out", "over", "same",
    "should", "similar", "similarities", "similarity", "some", "such", "than",
    "that", "the", "their", "them", "there", "these", "they", "this", "those",
    "through", "under", "very", "was", "were", "what", "when", "where",
    "which", "while", "who", "with", "would", "your",
  ]);
  const counts = new Map();
  for (const term of String(text || "").toLowerCase().match(/[a-z][a-z0-9]{2,}|[0-9]{2,}/g) || []) {
    if (!ignoredTerms.has(term)) counts.set(term, (counts.get(term) || 0) + 1);
  }
  return counts;
}

function compareSummaryToPassages(summary, passages) {
  const summaryTerms = getComparisonTerms(summary);
  const passageTerms = getComparisonTerms(passages.map(passage => passage.excerpt || "").join(" "));
  const mostFrequent = (source, predicate) =>
    [...source.entries()]
      .filter(([term]) => predicate(term))
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .slice(0, 6)
      .map(([term]) => term);
  return {
    shared: mostFrequent(summaryTerms, term => passageTerms.has(term)),
    summaryOnly: mostFrequent(summaryTerms, term => !passageTerms.has(term)),
    passageOnly: mostFrequent(passageTerms, term => !summaryTerms.has(term)),
  };
}

async function copyText(text, button) {
  await navigator.clipboard.writeText(text);
  const old = button.textContent;
  button.textContent = "Copied";
  setTimeout(() => { button.textContent = old; }, 1600);
}

function renderRetrievalAnalysis(container, result, heading) {
  const sources = result.sources || [];
  const analysis = result.analysis;
  const comparisons = new Map(
    (analysis.comparisons || []).map(comparison => [comparison.source_id, comparison])
  );
  const comparisonMarkup = sources.map(source => {
    const comparison = comparisons.get(source.source_id) || {};
    const terms = (items, fallback) => items.length
      ? items.map(escapeHtml).join(", ")
      : fallback;
    return `
      <li>
        <h4>[${escapeHtml(source.source_id)}] ${escapeHtml(source.case_name)}</h4>
        <p class="analysis-citation">${escapeHtml(source.decision_date)} · pages ${escapeHtml(source.pages)}${source.retrieval_similarity == null ? "" : ` · technical similarity ${Number(source.retrieval_similarity).toFixed(4)}`}</p>
        <ul>
          <li><strong>Exact terms in both:</strong> ${terms(comparison.shared_terms || [], "No exact keyword overlap identified.")}</li>
          <li><strong>Summary terms not in this excerpt:</strong> ${terms(comparison.summary_only_terms || [], "None identified.")}</li>
          <li><strong>Excerpt terms not in the summary:</strong> ${terms(comparison.passage_only_terms || [], "None identified.")}</li>
        </ul>
        <blockquote>${escapeHtml(source.excerpt)}</blockquote>
        <p><strong>Point to test:</strong> Check the full judgment and your evidence to see whether these passages apply or are distinguishable.</p>
      </li>
    `;
  }).join("");
  const copyContent = [
    `CASE SUMMARY\n${analysis.case_summary}`,
    ...sources.map(source => {
      const comparison = comparisons.get(source.source_id) || {};
      return `[${source.source_id}] ${source.case_name} · ${source.decision_date} · pages ${source.pages}\nShared terms: ${(comparison.shared_terms || []).join(", ") || "none identified"}\nSummary-only terms: ${(comparison.summary_only_terms || []).join(", ") || "none identified"}\nExcerpt-only terms: ${(comparison.passage_only_terms || []).join(", ") || "none identified"}\n${source.excerpt}`;
    }),
    `QUESTIONS FOR COUNSEL\n${analysis.review_questions.map(item => `- ${item}`).join("\n")}`,
    result.notice,
  ].join("\n\n");
  container.innerHTML = `
    <div class="case-analysis-header">
      <div>
        <span class="answer-kicker">Retrieval comparison · no language model</span>
        <h2>${escapeHtml(heading)}</h2>
      </div>
      <button class="copy-retrieval-analysis" type="button">Copy analysis</button>
    </div>
    <p class="analysis-notice">The summary below is user-provided and unverified. These are keyword comparisons and source passages—not generated legal arguments, findings of fact, or confirmed evidence gaps.</p>
    <section class="retrieval-case-summary"><h3>Case summary supplied</h3><p>${escapeHtml(analysis.case_summary)}</p></section>
    <section class="retrieval-comparisons"><h3>Retrieved passages to compare</h3><ol>${comparisonMarkup}</ol></section>
    <section class="retrieval-review">
      <h3>Questions to review with counsel</h3>
      <ul>${analysis.review_questions.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </section>
    <p class="analysis-notice">${escapeHtml(result.notice)}</p>
  `;
  const copyButton = container.querySelector(".copy-retrieval-analysis");
  copyButton.addEventListener("click", event =>
    copyText(copyContent, event.currentTarget)
  );
}

function renderResults(data) {
  const results = data.results || [];
  const cases = groupByCase(results);
  const references = cases.map((matchingCase, index) =>
    `[${index + 1}] ${matchingCase.case_name || matchingCase.file_name}; judgment date ${matchingCase.decision_date || "not available"}; pages ${[...matchingCase.pages].join(", ") || "not recorded"}; source ${matchingCase.file_name}; chunks ${matchingCase.passages.map(passage => passage.chunk_id).join(", ")}`
  ).join("\n");
  const excerpts = cases.map((matchingCase, index) =>
    `[${index + 1}] ${matchingCase.case_name || matchingCase.file_name}\nJudgment date: ${matchingCase.decision_date || "Not available in indexed metadata"}\n${matchingCase.passages.map(passage => `Pages ${passage.page_start}-${passage.page_end}: ${passage.excerpt}`).join("\n")}`
  ).join("\n\n");
  answerPanel.innerHTML = `
    <div class="results-heading">
      <div>
        <span class="answer-kicker">Potential matches · semantic search</span>
        <h2>${cases.length} potentially related case${cases.length === 1 ? "" : "s"}</h2>
      </div>
      <span class="result-count">${results.length} passage${results.length === 1 ? "" : "s"}</span>
    </div>
    <p class="searched-query"><strong>Your summary:</strong> ${escapeHtml(data.query)}</p>
    <p class="results-note">${escapeHtml(data.notice)} ${escapeHtml(data.candidates_retrieved)} passages were retrieved before duplicate removal (${escapeHtml(data.duplicates_removed)} duplicates removed). Similarity is a technical retrieval score, not a legal relevance or confidence rating.</p>
    ${cases.length ? `<ol class="case-results">${cases.map((matchingCase, index) => {
      const highlights = getPassageHighlights(matchingCase.passages, data.query);
      const pages = [...matchingCase.pages].sort((left, right) => Number(left) - Number(right));
      return `
        <li class="case-result">
          <article>
            <header class="case-result-header">
              <span class="case-rank">Case ${String.fromCharCode(65 + index)} · Potential match ${index + 1}</span>
              <span class="similarity-badge">Similarity ${matchingCase.retrieval_similarity == null ? "N/A" : Number(matchingCase.retrieval_similarity).toFixed(4)}</span>
            </header>
            <h3>${escapeHtml(matchingCase.case_name || "Case name unavailable in metadata")}</h3>
            <ul class="case-facts">
              <li><strong>Judgment date:</strong> ${escapeHtml(matchingCase.decision_date || "Not available in indexed metadata")}</li>
              <li><strong>Relevant pages:</strong> ${escapeHtml(pages.length ? pages.join(", ") : "Not recorded")}</li>
              <li><strong>Matching passages:</strong> ${matchingCase.passages.length}</li>
              <li><strong>Source document:</strong> ${escapeHtml(matchingCase.file_name || "Not available")}</li>
            </ul>
            <div class="case-highlights">
              <h4>Relevant retrieved text <span>(verbatim excerpts)</span></h4>
              ${highlights.length
                ? `<ul>${highlights.map(highlight => `<li><blockquote>${escapeHtml(highlight.text)}</blockquote></li>`).join("")}</ul>`
                : "<p>No readable excerpt was available for this result.</p>"}
            </div>
            <details class="full-source">
              <summary>View full retrieved judgment text · ${matchingCase.passages.length} passage${matchingCase.passages.length === 1 ? "" : "s"}</summary>
              <ul>
                ${matchingCase.passages.map(passage => `
                  <li>
                    <strong>Pages ${escapeHtml(passage.page_start)}–${escapeHtml(passage.page_end)}</strong>
                    <span class="source-score">Technical similarity: ${passage.retrieval_similarity == null ? "N/A" : Number(passage.retrieval_similarity).toFixed(4)} · chunk ${escapeHtml(passage.chunk_id)}</span>
                    <blockquote>${escapeHtml(passage.excerpt)}</blockquote>
                  </li>
                `).join("")}
              </ul>
            </details>
          </article>
        </li>
      `;
    }).join("")}</ol>` : `<div class="no-results"><h3>No matching passages retrieved</h3><p>${data.corpus_available === false
      ? "This demo deployment does not include the indexed judgment corpus, so case search is unavailable."
      : "This does not establish that no relevant judgment or law exists. Try a different summary or legal issue."}</p></div>`}
    ${cases.length ? `
      <section class="research-tool-card" aria-labelledby="caseChatActionTitle">
        <div class="research-tool-copy">
          <span class="research-tool-label">01 · Case chatbot</span>
          <h2 id="caseChatActionTitle">Have questions about these cases?</h2>
          <p>Ask follow-up questions, compare retrieved passages, and review source-grounded questions.</p>
        </div>
        <button id="openCaseChat" type="button" aria-controls="caseChat" aria-expanded="false">Ask about these cases</button>
      </section>
      <section id="caseChat" class="case-chat" aria-labelledby="caseChatTitle" hidden>
        <div class="case-chat-heading">
          <div>
            <span class="answer-kicker">Retrieval-only chat</span>
            <h2 id="caseChatTitle">Ask about these judgments</h2>
          </div>
        </div>
        <p class="chat-intro">Search judgments by meaning and see the most relevant source sentences. Select a case or say “Case A” to keep the answer focused. No language model is used.</p>
        <div class="chat-focus">
          <label for="caseChatFocus">Focus on a judgment</label>
          <select id="caseChatFocus">
            <option value="">All ${cases.length} retrieved judgments</option>
            ${cases.map((matchingCase, index) => `
              <option value="${index}" ${matchingCase.document_id ? "" : "disabled"}>Case ${String.fromCharCode(65 + index)} · ${escapeHtml(matchingCase.case_name || matchingCase.file_name || "Case name unavailable")}</option>
            `).join("")}
          </select>
        </div>
        <div id="chatThread" class="chat-thread" role="log" aria-live="polite" aria-relevant="additions text">
          <p class="chat-welcome">For example: “What does this judgment say about Article 21?”</p>
        </div>
        <form id="caseChatForm" class="chat-form">
          <label class="sr-only" for="caseChatInput">Ask about the retrieved judgments</label>
          <textarea id="caseChatInput" rows="2" maxlength="1000" placeholder="Ask a question about these judgments…" required></textarea>
          <button type="submit">Search judgments</button>
        </form>
      </section>
      <section class="research-tool-card analysis-tool-card" aria-labelledby="analysisActionTitle">
        <div class="research-tool-copy">
          <span class="research-tool-label">02 · Lawyer research</span>
          <h2 id="analysisActionTitle">Compare this case with precedents</h2>
          <p>Review retrieved passages, exact keyword overlap, and questions to check against your case facts and evidence.</p>
        </div>
        <button id="analyzeCase" type="button">Analyze this case</button>
      </section>
      <section id="caseAnalysisPanel" class="case-analysis-panel" aria-live="polite" hidden></section>
    ` : ""}
    <p class="results-note">Case outcomes, holdings, applicable laws, and evidence gaps are not inferred by this retrieval-only search. Review the full source judgments for context.</p>
    <ul class="limitations-list">${(data.limitations || []).map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    <p class="disclaimer">${escapeHtml(data.disclaimer)}</p>
    <div class="copy-row">
      <button id="copyExcerpts" ${cases.length ? "" : "disabled"}>Copy case bullets</button>
      <button id="copyReferences" ${cases.length ? "" : "disabled"}>Copy source references</button>
    </div>
  `;
  document.querySelector("#copyExcerpts").addEventListener("click", event => copyText(excerpts, event.currentTarget));
  document.querySelector("#copyReferences").addEventListener("click", event => copyText(references, event.currentTarget));

  const analyzeButton = document.querySelector("#analyzeCase");
  if (analyzeButton) {
    const analysisPanel = document.querySelector("#caseAnalysisPanel");
    analyzeButton.addEventListener("click", async () => {
      analyzeButton.disabled = true;
      analyzeButton.textContent = "Comparing passages…";
      analysisPanel.hidden = false;
      analysisPanel.innerHTML = '<p class="analysis-notice">Retrieving and comparing selected judgment passages…</p>';
      try {
        const documentIds = cases.map(matchingCase => matchingCase.document_id).filter(Boolean);
        const response = await fetch("/api/analyze-case", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            case_summary: data.query,
            question: "Find judgment passages relevant to these case facts and identify points to compare.",
            document_ids: documentIds,
          })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Case comparison failed");
        renderRetrievalAnalysis(analysisPanel, result, "Case-to-precedent comparison");
      } catch (error) {
        analysisPanel.innerHTML = `<p class="chat-error">Unable to analyze this case: ${escapeHtml(error.message)}</p>`;
      } finally {
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze this case";
        analysisPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  }

  const chatForm = document.querySelector("#caseChatForm");
  if (chatForm) {
    const openChatButton = document.querySelector("#openCaseChat");
    const chatPanel = document.querySelector("#caseChat");
    const chatInput = document.querySelector("#caseChatInput");
    const focusSelect = document.querySelector("#caseChatFocus");
    const chatThread = document.querySelector("#chatThread");
    const chatSubmit = chatForm.querySelector("button");
    const chatHistory = [];
    openChatButton.addEventListener("click", () => {
      chatPanel.hidden = false;
      openChatButton.setAttribute("aria-expanded", "true");
      openChatButton.textContent = "Chat about these cases";
      chatPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      chatInput.focus({ preventScroll: true });
    });
    chatForm.addEventListener("submit", async event => {
      event.preventDefault();
      const question = chatInput.value.trim();
      if (!question) { chatInput.focus(); return; }
      const caseAnalysisRequest = /\b(argument|arguments|argue|argues|draft|analy[sz]e|analysis|assess|assessment|evaluate|evaluation|strategy|submissions?)\b/i.test(question);
      const comparisonRequest = /\b(compare|comparison|contrast|difference|differences|similarity|similarities)\b/i.test(question);
      const evidenceGapRequest = /\b(evidence gap|evidence gaps|missing evidence|evidence missing)\b/i.test(question);

      const userMessage = document.createElement("div");
      userMessage.className = "chat-message user";
      userMessage.textContent = question;
      chatThread.append(userMessage);

      const assistantMessage = document.createElement("article");
      assistantMessage.className = "chat-message assistant";
      assistantMessage.innerHTML = "<p>Searching the selected judgments for the closest passages…</p>";
      chatThread.append(assistantMessage);
      chatInput.value = "";
      chatSubmit.disabled = true;
      chatSubmit.textContent = "Searching…";
      assistantMessage.scrollIntoView({ block: "nearest" });

      const selectedCase = focusSelect.value === "" ? null : cases[Number(focusSelect.value)];
      const requestedCases = selectedCase ? [selectedCase] : findCasesMentioned(question, cases);
      const searchCases = requestedCases.length ? requestedCases : cases;
      try {
        const recentQuestions = chatHistory.slice(-2);
        chatHistory.push(question);
        if (chatHistory.length > 4) chatHistory.shift();
        const questionForSearch = removeCaseNamesFromQuestion(question, searchCases);
        const retrievalQuery = [
          `Current question: ${questionForSearch || question}`,
          ...recentQuestions.map(item =>
            `Previous question: ${removeCaseNamesFromQuestion(item, searchCases).slice(0, 250)}`
          ),
          `Original case summary: ${String(data.query).slice(0, 2400)}`,
        ].join("\n\n").slice(0, 4000);

        const response = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: retrievalQuery,
            top_k: 8,
            document_ids: searchCases.map(matchingCase => matchingCase.document_id).filter(Boolean),
          })
        });
        const answer = await response.json();
        if (!response.ok) throw new Error(answer.detail || "Judgment search failed");

        const retrievedCases = groupByCase(answer.results || []);
        const answerCases = (requestedCases.length
          ? retrievedCases.filter(retrievedCase =>
              requestedCases.some(requestedCase => requestedCase.document_id === retrievedCase.document_id)
            )
          : retrievedCases
        ).slice(0, 3);
        const showComparison = comparisonRequest;
        const showEvidenceReview = evidenceGapRequest;
        const answerIntro = caseAnalysisRequest
          ? "I can’t draft legal arguments without a language model. These are the most relevant passages in the selected judgments to research against your facts."
          : comparisonRequest
            ? "These are the closest retrieved passages to compare with your case summary. Word overlap is only a search aid."
            : evidenceGapRequest
              ? "These passages may help you review evidence issues. They do not establish that evidence is missing."
              : "Here are the passages that best match your question in the selected judgments.";
        assistantMessage.innerHTML = answerCases.length
          ? `<p>${escapeHtml(answerIntro)}</p>
            <ol class="chat-sources">${answerCases.map((matchingCase, index) => {
              const highlights = getPassageHighlights(matchingCase.passages, questionForSearch, 2);
              const caseIndex = cases.findIndex(item =>
                item.document_id === matchingCase.document_id
              );
              const caseLabel = caseIndex < 0
                ? `Case ${String.fromCharCode(65 + index)}`
                : `Case ${String.fromCharCode(65 + caseIndex)}`;
              return `
                <li>
                  <strong>[${caseLabel}] ${escapeHtml(matchingCase.case_name || matchingCase.file_name || "Case name unavailable")}</strong>
                  <span>Judgment: ${escapeHtml(matchingCase.decision_date || "not available")}</span>
                  <ul class="chat-highlights">${highlights.map(highlight => {
                    const sourcePassage = matchingCase.passages[highlight.passageIndex] || {};
                    const pageStart = sourcePassage.page_start;
                    const pageEnd = sourcePassage.page_end;
                    const pageReference = pageStart == null
                      ? "pages not recorded"
                      : `page${pageEnd != null && pageEnd !== pageStart ? "s" : ""} ${pageStart}${pageEnd != null && pageEnd !== pageStart ? `–${pageEnd}` : ""}`;
                    return `
                      <li>
                        <blockquote>${escapeHtml(highlight.text || "No readable sentence was available.")}</blockquote>
                        <small>Source: ${escapeHtml(pageReference)} · chunk ${escapeHtml(sourcePassage.chunk_id || "not recorded")}<br>${highlight.matchedTerms.length
                          ? `Exact question terms found: ${escapeHtml(highlight.matchedTerms.join(", "))}`
                          : "Retrieved by semantic similarity; no exact question terms appear in this sentence."}</small>
                      </li>
                    `;
                  }).join("")}</ul>
                </li>
              `;
            }).join("")}</ol>
              ${showComparison ? `<section class="chat-analysis">
                <h4>Similarities in the summary and retrieved passages</h4>
                <ul>${answerCases.map(matchingCase => {
                  const comparison = compareSummaryToPassages(data.query, matchingCase.passages);
                  return `<li><strong>${escapeHtml(matchingCase.case_name || matchingCase.file_name || "Case name unavailable")}</strong><ul><li>Shared terms: ${escapeHtml(comparison.shared.join(", ") || "No exact keyword overlap identified in these excerpts.")}</li><li>Summary terms not found in these excerpts: ${escapeHtml(comparison.summaryOnly.join(", ") || "None identified.")}</li><li>Terms in these excerpts not found in the summary: ${escapeHtml(comparison.passageOnly.join(", ") || "None identified.")}</li></ul></li>`;
                }).join("")}</ul>
                <p>These are keyword-level matches and differences, not findings about the full facts or legal effect of either case.</p>
              </section>` : ""}
              ${showEvidenceReview ? `<section class="chat-analysis evidence-checklist">
                <h4>Evidence and facts to verify</h4>
                <p>Use the cited judgment passages to check:</p>
                <ul><li>What material facts are supported, disputed, or still unverified in your record.</li><li>Whether the full judgment’s legal test fits your facts and procedural stage.</li><li>Which contrary evidence or alternative account must be addressed.</li></ul>
                <p>These are review prompts, not findings that evidence is missing.</p>
              </section>` : ""}
              <p class="chat-caveat">These are source passages, not a generated answer or a legal conclusion. Check the cited pages in the full judgment.</p>`
          : "<p>No matching passages were retrieved for that question. This does not establish that no relevant information exists in the full corpus.</p>";
      } catch (error) {
        assistantMessage.innerHTML = `<p class="chat-error">${caseAnalysisRequest ? "Unable to compare the case with precedents" : "Unable to search the judgments"}: ${escapeHtml(error.message)}</p>`;
      } finally {
        chatSubmit.disabled = false;
        chatSubmit.textContent = "Search judgments";
        assistantMessage.scrollIntoView({ block: "nearest" });
        chatInput.focus();
      }
    });
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) { input.focus(); return; }
  submit.disabled = true;
  submit.innerHTML = "Finding cases <span>…</span>";
  answerPanel.innerHTML = '<div class="loading-line"></div><div class="empty-state"><h2>Finding related judgments</h2><p>Searching the local corpus and preparing concise case details…</p></div>';
  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5 })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Research request failed");
    renderResults(data);
  } catch (error) {
    answerPanel.innerHTML = `<span class="answer-kicker">Search error</span><h2>Unable to find cases</h2><p>${escapeHtml(error.message)}</p>`;
  } finally {
    submit.disabled = false;
    submit.innerHTML = "Find similar cases <span>→</span>";
  }
});

checkStatus();
