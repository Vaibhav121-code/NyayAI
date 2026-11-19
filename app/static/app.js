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
    const data = await response.json();
    status.className = "status ok";
    status.innerHTML = `<i></i> ${data.mode === "retrieval_only" ? "Local retrieval ready" : "System ready"}`;
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

function getPassageHighlights(passages, query) {
  const ignoredTerms = new Set([
    "and", "are", "can", "does", "for", "from", "how", "into", "may", "that",
    "the", "this", "under", "what", "when", "where", "which", "with",
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
      score: [...terms].reduce((score, term) => score + Number(sentenceTerms.has(term)), 0),
    };
  });
  const matching = ranked.filter(sentence => sentence.score > 0);
  const selected = (matching.length ? matching : ranked).slice()
    .sort((left, right) =>
      right.score - left.score ||
      left.passageIndex - right.passageIndex ||
      left.sentenceIndex - right.sentenceIndex
    )
    .slice(0, 3)
    .sort((left, right) =>
      left.passageIndex - right.passageIndex ||
      left.sentenceIndex - right.sentenceIndex
    );
  return selected.map(sentence => ({
    text: sentence.text.length > 420 ? `${sentence.text.slice(0, 417).trimEnd()}…` : sentence.text,
    passageIndex: sentence.passageIndex,
  }));
}

async function copyText(text, button) {
  await navigator.clipboard.writeText(text);
  const old = button.textContent;
  button.textContent = "Copied";
  setTimeout(() => { button.textContent = old; }, 1600);
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
              <span class="case-rank">Potential match ${index + 1}</span>
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
    }).join("")}</ol>` : '<div class="no-results"><h3>No matching passages retrieved</h3><p>This does not establish that no relevant judgment or law exists. Try a different summary or legal issue.</p></div>'}
    ${cases.length ? `
      <div class="argument-launch">
        <div>
          <strong>Need to prepare for court?</strong>
          <p>Build a source-cited outline from these passages to review with a lawyer.</p>
        </div>
        <button id="prepareArguments" type="button">Prepare argument outline</button>
      </div>
      <section id="argumentOutline" class="argument-outline" aria-live="polite" hidden></section>
    ` : ""}
    ${cases.length ? `
      <section class="case-chat" aria-labelledby="caseChatTitle">
        <div class="case-chat-heading">
          <div>
            <span class="answer-kicker">Retrieval-only chat</span>
            <h2 id="caseChatTitle">Ask about these judgments</h2>
          </div>
        </div>
        <p class="chat-intro">Ask a follow-up about a case, legal issue, or passage. Each reply searches only the judgments shown above and returns source excerpts, not generated legal analysis.</p>
        <div id="chatThread" class="chat-thread" role="log" aria-live="polite" aria-relevant="additions text">
          <p class="chat-welcome">For example: “What does this judgment say about Article 21?”</p>
        </div>
        <form id="caseChatForm" class="chat-form">
          <label class="sr-only" for="caseChatInput">Ask about the retrieved judgments</label>
          <textarea id="caseChatInput" rows="2" maxlength="1000" placeholder="Ask a question about these judgments…" required></textarea>
          <button type="submit">Search judgments</button>
        </form>
      </section>
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

  const argumentButton = document.querySelector("#prepareArguments");
  if (argumentButton) {
    const outlinePanel = document.querySelector("#argumentOutline");
    argumentButton.addEventListener("click", () => {
      const points = cases.flatMap(matchingCase =>
        getPassageHighlights(matchingCase.passages, data.query)
          .slice(0, 2)
          .map(highlight => ({ matchingCase, highlight }))
      );
      const outlineText = [
        "SOURCE-CITED ARGUMENT RESEARCH OUTLINE — REVIEW WITH QUALIFIED COUNSEL",
        `Issue or facts provided: ${data.query}`,
        "",
        ...points.flatMap(({ matchingCase, highlight }, index) => {
          const passage = matchingCase.passages[highlight.passageIndex];
          const pages = [passage.page_start, passage.page_end]
            .filter(page => page != null)
            .join("–") || "not recorded";
          return [
            `${index + 1}. ${matchingCase.case_name || matchingCase.file_name || "Case name unavailable"}`,
            `   Judgment date: ${matchingCase.decision_date || "not available"}; pages: ${pages}`,
            `   Retrieved passage: “${highlight.text}”`,
            "   Application to established facts: [Explain the factual connection or distinction.]",
            "   Counterarguments: [Check contrary authority and relevant distinctions.]",
            "",
          ];
        }),
        "Verify each quotation in the full judgment. This outline does not assess whether an argument is legally suitable or applicable.",
      ].join("\n");

      outlinePanel.innerHTML = `
        <div class="argument-outline-header">
          <div>
            <span class="answer-kicker">Research aid · source-cited</span>
            <h2>Argument outline to review</h2>
          </div>
          <button id="copyArgumentOutline" type="button">Copy outline</button>
        </div>
        <p class="argument-caveat">This is a research scaffold, not a ready-to-file submission. The retrieval system cannot determine which arguments apply, confirm that a passage is the court’s holding, or identify every adverse authority. Verify the full judgment and review with qualified counsel.</p>
        <p class="argument-issue"><strong>Your issue or facts:</strong> ${escapeHtml(data.query)}</p>
        <ol class="argument-points">${points.map(({ matchingCase, highlight }) => {
          const passage = matchingCase.passages[highlight.passageIndex];
          const pages = [passage.page_start, passage.page_end]
            .filter(page => page != null)
            .join("–") || "Not recorded";
          return `
            <li>
              <h3>${escapeHtml(matchingCase.case_name || matchingCase.file_name || "Case name unavailable")}</h3>
              <p class="argument-citation">Judgment date: ${escapeHtml(matchingCase.decision_date || "Not available")} · pages ${escapeHtml(pages)}</p>
              <blockquote>${escapeHtml(highlight.text)}</blockquote>
              <p><strong>Application to assess:</strong> Explain how established facts may support or distinguish this passage.</p>
              <p><strong>Check before presenting:</strong> Review the passage in context and identify contrary authorities or factual distinctions.</p>
            </li>
          `;
        }).join("")}</ol>
        <p class="argument-caveat">Similarity scores only reflect technical retrieval, not legal relevance or strength.</p>
      `;
      outlinePanel.hidden = false;
      argumentButton.textContent = "Refresh argument outline";
      document.querySelector("#copyArgumentOutline").addEventListener("click", event =>
        copyText(outlineText, event.currentTarget)
      );
      outlinePanel.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  const chatForm = document.querySelector("#caseChatForm");
  if (chatForm) {
    const chatInput = document.querySelector("#caseChatInput");
    const chatThread = document.querySelector("#chatThread");
    const chatSubmit = chatForm.querySelector("button");
    chatForm.addEventListener("submit", async event => {
      event.preventDefault();
      const question = chatInput.value.trim();
      if (!question) { chatInput.focus(); return; }

      const userMessage = document.createElement("div");
      userMessage.className = "chat-message user";
      userMessage.textContent = question;
      chatThread.append(userMessage);

      const assistantMessage = document.createElement("article");
      assistantMessage.className = "chat-message assistant";
      assistantMessage.innerHTML = "<p>Searching the judgment corpus for relevant passages…</p>";
      chatThread.append(assistantMessage);
      chatInput.value = "";
      chatSubmit.disabled = true;
      assistantMessage.scrollIntoView({ block: "nearest" });

      const caseNames = cases.map(matchingCase => matchingCase.case_name).filter(Boolean).join("; ");
      const context = `Initial case summary: ${data.query}\nInitially retrieved cases: ${caseNames}`.slice(0, 2800);
      try {
        const response = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: `${question}\n\n${context}`,
            top_k: 5,
            document_ids: cases.map(matchingCase => matchingCase.document_id).filter(Boolean),
          })
        });
        const answer = await response.json();
        if (!response.ok) throw new Error(answer.detail || "Judgment search failed");

        const answerCases = groupByCase(answer.results || []);
        assistantMessage.innerHTML = answerCases.length
          ? `<p>Here are retrieved passages related to your question. Review them in the context of each full judgment.</p>
            <ol class="chat-sources">${answerCases.map(matchingCase => `
              <li>
                <strong>${escapeHtml(matchingCase.case_name || matchingCase.file_name || "Case name unavailable")}</strong>
                <span>Judgment: ${escapeHtml(matchingCase.decision_date || "not available")} · pages ${escapeHtml([...matchingCase.pages].join(", ") || "not recorded")}</span>
                ${matchingCase.passages.map(passage => `<blockquote>${escapeHtml(passage.excerpt || "No readable passage available.")}</blockquote>`).join("")}
              </li>
            `).join("")}</ol>
            <p class="chat-caveat">These are source excerpts, not a generated answer or a legal conclusion.</p>`
          : "<p>No matching passages were retrieved for that question. This does not establish that no relevant information exists in the full corpus.</p>";
      } catch (error) {
        assistantMessage.innerHTML = `<p class="chat-error">Unable to search the judgments: ${escapeHtml(error.message)}</p>`;
      } finally {
        chatSubmit.disabled = false;
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
