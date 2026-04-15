// Initialize Mermaid diagrams for MkDocs
document.addEventListener("DOMContentLoaded", function () {
  if (typeof mermaid === "undefined") {
    return;
  }

  mermaid.initialize({
    startOnLoad: false,
    theme: "default",
    flowchart: { useMaxWidth: true, htmlLabels: true },
  });

  function processAndRender() {
    var codeBlocks = document.querySelectorAll("pre code");
    var keywords = [
      "graph ",
      "flowchart ",
      "sequenceDiagram",
      "classDiagram",
      "stateDiagram",
      "erDiagram",
      "gantt",
      "pie",
    ];

    codeBlocks.forEach(function (block) {
      var content = block.textContent.trim();
      var isMermaid = keywords.some(function (kw) {
        return content.startsWith(kw);
      });
      if (!isMermaid) return;

      var container = document.createElement("div");
      container.className = "mermaid";
      container.textContent = content;
      block.parentNode.parentNode.replaceChild(container, block.parentNode);
    });

    var elements = document.querySelectorAll(".mermaid:not([data-processed])");
    if (elements.length > 0) {
      try {
        mermaid.init(undefined, elements);
      } catch (e) {
        console.error("Mermaid rendering error:", e);
      }
    }
  }

  processAndRender();

  // Re-process after MkDocs page navigation
  var observer = new MutationObserver(function (mutations) {
    for (var i = 0; i < mutations.length; i++) {
      if (mutations[i].addedNodes.length > 0) {
        setTimeout(processAndRender, 100);
        return;
      }
    }
  });

  var content =
    document.querySelector(".rst-content") ||
    document.querySelector("main") ||
    document.body;
  observer.observe(content, { childList: true, subtree: true });
});
