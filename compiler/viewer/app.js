const rawOutput = document.querySelector("#raw-output");
const fileInput = document.querySelector("#file-input");
const parseButton = document.querySelector("#parse-button");
const clearButton = document.querySelector("#clear-button");
const parseStatus = document.querySelector("#parse-status");
const variantsEl = document.querySelector("#variants");
const variantTemplate = document.querySelector("#variant-template");
const variantCount = document.querySelector("#variant-count");
const guardCount = document.querySelector("#guard-count");
const codeLineCount = document.querySelector("#code-line-count");

function splitGuardBlocks(rawGuards) {
  const blocks = [];
  const matches = [...rawGuards.matchAll(/^Name: .*$/gm)];

  if (matches.length === 0) {
    return rawGuards.trim() ? [rawGuards.trim()] : [];
  }

  for (let idx = 0; idx < matches.length; idx += 1) {
    const start = matches[idx].index;
    const end = idx + 1 < matches.length ? matches[idx + 1].index : rawGuards.length;
    const block = rawGuards.slice(start, end).trim();
    if (block) blocks.push(block);
  }

  return blocks;
}

function readField(block, label) {
  const pattern = new RegExp(`^\\s*${label}:\\s*(.*)$`, "m");
  const match = block.match(pattern);
  return match ? match[1].trim() : "";
}

function guardLabel(block, idx) {
  const name = readField(block, "Name");
  const source = readField(block, "Source");
  const createFunction = readField(block, "Create Function");
  const parts = [`#${idx + 1}`];

  if (createFunction) parts.push(createFunction);
  if (source) parts.push(source);
  if (name && name !== "''") parts.push(name);

  return parts.join(" · ");
}

function parseVisualizerOutput(text) {
  const marker = "new variant!";
  const sections = text.split(marker).slice(1);

  return sections.map((section, idx) => {
    const guardMarker = "==GUARDS==";
    const guardStart = section.indexOf(guardMarker);
    const code = guardStart >= 0 ? section.slice(0, guardStart).trim() : section.trim();
    const guardSection = guardStart >= 0 ? section.slice(guardStart + guardMarker.length).trim() : "";
    const lines = guardSection.split(/\r?\n/);
    const statedGuardCount = Number.parseInt(lines[0], 10);
    const rawGuards = Number.isFinite(statedGuardCount) ? lines.slice(1).join("\n") : guardSection;
    const guards = splitGuardBlocks(rawGuards);

    return {
      id: idx,
      code,
      guards,
      statedGuardCount: Number.isFinite(statedGuardCount) ? statedGuardCount : guards.length,
    };
  });
}

function setSummary(variants) {
  const totalGuards = variants.reduce((total, variant) => total + variant.guards.length, 0);
  const totalCodeLines = variants.reduce((total, variant) => {
    if (!variant.code) return total;
    return total + variant.code.split(/\r?\n/).length;
  }, 0);

  variantCount.textContent = String(variants.length);
  guardCount.textContent = String(totalGuards);
  codeLineCount.textContent = String(totalCodeLines);
}

function renderEmpty(message) {
  variantsEl.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "empty";
  empty.textContent = message;
  variantsEl.append(empty);
}

function renderVariant(variant) {
  const node = variantTemplate.content.firstElementChild.cloneNode(true);
  const summary = node.querySelector(".variant-summary");
  const body = node.querySelector(".variant-body");
  const title = node.querySelector(".variant-title");
  const meta = node.querySelector(".variant-meta");
  const guardSelect = node.querySelector(".guard-select");
  const guardDetail = node.querySelector(".guard-detail");
  const codeToggle = node.querySelector(".code-toggle");
  const codeBlock = node.querySelector(".code-block");
  const codeLines = variant.code ? variant.code.split(/\r?\n/).length : 0;

  title.textContent = `Variant ${variant.id + 1}`;
  meta.textContent = `${variant.guards.length}/${variant.statedGuardCount} guards parsed · ${codeLines} code lines`;
  codeBlock.textContent = variant.code || "No FX code found.";

  if (variant.guards.length === 0) {
    const option = document.createElement("option");
    option.textContent = "No guards found";
    guardSelect.append(option);
    guardDetail.textContent = "No guard block was found for this variant.";
  } else {
    variant.guards.forEach((guard, idx) => {
      const option = document.createElement("option");
      option.value = String(idx);
      option.textContent = guardLabel(guard, idx);
      guardSelect.append(option);
    });
    guardDetail.textContent = variant.guards[0];
  }

  summary.addEventListener("click", () => {
    const expanded = summary.getAttribute("aria-expanded") === "true";
    summary.setAttribute("aria-expanded", String(!expanded));
    body.hidden = expanded;
  });

  guardSelect.addEventListener("change", () => {
    const selected = Number.parseInt(guardSelect.value, 10);
    guardDetail.textContent = variant.guards[selected] || "No guard selected.";
  });

  codeToggle.addEventListener("click", () => {
    const expanded = codeToggle.getAttribute("aria-expanded") === "true";
    codeToggle.setAttribute("aria-expanded", String(!expanded));
    codeToggle.textContent = expanded ? "Show Code" : "Hide Code";
    codeBlock.hidden = expanded;
  });

  return node;
}

function render(variants) {
  setSummary(variants);
  variantsEl.replaceChildren();

  if (variants.length === 0) {
    renderEmpty("No variants parsed.");
    return;
  }

  variants.forEach((variant) => {
    variantsEl.append(renderVariant(variant));
  });
}

function parseCurrentText() {
  const variants = parseVisualizerOutput(rawOutput.value);
  render(variants);
  parseStatus.textContent = variants.length === 1 ? "Parsed 1 variant." : `Parsed ${variants.length} variants.`;
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;

  rawOutput.value = await file.text();
  parseCurrentText();
});

parseButton.addEventListener("click", parseCurrentText);

clearButton.addEventListener("click", () => {
  rawOutput.value = "";
  parseStatus.textContent = "";
  render([]);
});

render([]);
